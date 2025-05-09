import torch
import torch_pruning as tp
import torch.nn as nn
import copy

# 本地预训练模型
weights_path = "cft_best_llvip.pt"

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(weights_path, map_location=device)  # 加载 checkpoint
model = ckpt['model']  # 提取 PyTorch 模型
model = model.float()  # 处理模型

# print(model)
with open("cft_model_structure.txt", "w") as f:
    f.write(str(model))

# 1、设置输入张量的示例
torch.manual_seed(0)  # 固定随机种子
img_rgb = torch.randn(8, 3, 640, 640).to(device)  # RGB 图像输入
img_ir = torch.randn(8, 3, 640, 640).to(device)   # 红外图像输入
example_inputs=[img_rgb, img_ir]

# 2、忽略剪枝的层（比如最后的分类层或输出层），顺便给所有层加上需要梯度
ignored_layers_base = []

from models.yolo_test import Detect

for m in model.modules():
    m.requires_grad_(True)
    if isinstance(m, Detect):
        ignored_layers_base.append(m)

print('---------ignored_layers_base-----------')
print(ignored_layers_base)
print('---------ignored_layers_base-----------')

# 定义重要性度量标准
imp = tp.importance.GroupNormImportance(p=2)

# 计算剪枝前的模型计算量和参数量
base_macs, base_nparams = tp.utils.count_ops_and_params(model,example_inputs)
base_macs_G = base_macs / 1e9  # 转换为G
base_nparams_M = base_nparams / 1e6  # 转换为M
print(f"原始模型 MACs: {base_macs_G:.2f}G, 参数量: {base_nparams_M:.2f}M")

################################## 剪枝特殊层 ###################

# 判断当前层是否在某一层
# 注意要加上'.'，
def is_in_stream(name,stream):
    module_prefix = ".".join(name.split(".")[:2])  # 获取model.x
    # print(f"module_prefix:{module_prefix} ")
    return any(module_prefix == prefix for prefix in stream)

# 1、剪枝RGB层
# Stream1 modules
stream1 = [f"model.{i}" for i in range(0, 8)]

# 构造 ignored_layers（排除所有不属于 Stream1 的模块）
ignored_layers = [] 
ignored_layers.append(ignored_layers_base[0]) # detect总是要忽略的,注意不能直接赋值

for name, module in model.named_modules():
    # 整体的model不能被屏蔽
    if(name == 'model' or name == ''):
        continue

    if not is_in_stream(name,stream1):  # 不在 Stream1 的都忽略
        print(f"name:{name}")
        ignored_layers.append(module)

print(len(ignored_layers))
# print(ignored_layers)

# 创建 pruner
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs,
    global_pruning=True,
    importance=imp,
    pruning_ratio=0.5,
    ignored_layers=ignored_layers,
)

# # 4、交互式剪枝
# for i,group in enumerate(pruner.step(interactive=True)): # Warning: groups must be handled sequentially. Do not keep them as a list.

#     # # do whatever you like with the group 
#     # dep, idxs = group[0] # get the idxs
#     # target_module = dep.target.module # get the root module
#     # pruning_fn = dep.handler # get the pruning function

#     group.prune()
#     print(str(i)+"\n"+str(group))

pruner.step()


################################## 剪枝特殊层 ######################




# 计算剪枝后参数量
macs, nparams = tp.utils.count_ops_and_params(model,example_inputs)
macs_G = macs / 1e9  # 转换为G
nparams_M = nparams / 1e6  # 转换为M
print(f"MACs 剩余: {macs_G:.2f}G ({macs/base_macs:.4f},{1-macs/base_macs:.4f}), 参数剩余: {nparams_M:.2f}M ({nparams/base_nparams:.4f},{1-nparams/base_nparams:.4f})")


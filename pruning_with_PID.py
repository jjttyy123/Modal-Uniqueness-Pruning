# 直接使用ignore会有一些问题，和直接使用0.5剪枝率的torch-pruning实际剪枝率为0.7357(参数)，0.7329（MACs）
# 但是使用ignore的则是0.6777(参数)和0.7368(MACs)。这说明有一些参数因为ignore的问题被忽略一直没有被剪
# 这里我们直接分配剪枝率给指定的层，按照我们经过softmax算得的结果
# 现在我们平均分配50%的结果是0.7402(参数),0.7278(MACs)

import torch
import torch.nn as nn
import torch_pruning as tp
import copy
from utils.parameters_set_zero import zero_out_model_params, zero_out_specific_modules,analyze_output, check_zeroed_params, check_model_output_is_zero  
from utils.yolo_branch_analyzer import YoloStreamAnalyzer
from utils.model_eval import multimodal_eval,eval_all_combination,json_to_word
import json
from utils.loss import ComputeLoss
from train_for_pruning import start_train # 导入训练
import numpy as np
import gc 


############################  一、双分支分析 ##########################
# 1、加载模型和输入
device = torch.device('cpu')
# ckpt = torch.load("cft_best_m3fd.pt", map_location=device)
ckpt = torch.load("cft_best_flir.pt", map_location=device)
model = ckpt['model'].float().to(device)
torch.manual_seed(0)  # 固定随机种子
example_inputs = [torch.randn(8, 3, 640, 640).to(device), torch.randn(8, 3, 640, 640).to(device)]

for m in model.modules():
    m.requires_grad_(True)

# 2、创建分析器对象，后面的操作为是否需要移除对应的ElementWiseOp类型的节点，减少展示的节点数量
analyzer = YoloStreamAnalyzer(model, example_inputs, enable_elementwise_removal=True)

# 3、获取分析结果
# 3.1 获取stream1、stream2和shared_nodes
stream1, stream2, shared_nodes = analyzer.analyze()

# print("Stream 1:", [str(node) for node in stream1])
# print("Stream 2:", [str(node) for node in stream2])
# print("Shared Nodes:", [str(node) for node in shared_nodes])

# 3.2 绘制依赖图
# analyzer.plot_dependency_graph(stream1, stream2, shared_nodes, filename="yolo_dependency_graph.png")

# 4、用于存储每个流的父级模块路径
# 4.1 构造存储结构set
stream1_modules = set()
stream2_modules = set()
shared_modules = set()

# 4.2 记录父级模块路径
def get_parent_module_path(stream_module,stream):
    # 遍历stream，记录其中的父级模块路径
    for node in stream:
        # 提取模块的路径，例如从"model.model.4.cv3.bn"中提取出"model.4"
        parent_module_path = ".".join([part for part in node.path.split(".")[1:3] if part])  # 去掉 model.model 前缀，只保留 model.3
        # 只记录有效的路径，忽略'<unknown>'
        if parent_module_path != "<unknown>" and parent_module_path != "":
            stream_module.add(parent_module_path)

get_parent_module_path(stream1_modules,stream1)
get_parent_module_path(stream2_modules,stream2)
get_parent_module_path(shared_modules,shared_nodes)

# 4.3 输出stream1、stream2和共享节点的父级模块路径
print("Stream1 Modules: ", sorted(stream1_modules))
print("Stream2 Modules: ", sorted(stream2_modules))
print("Shared Modules: ", sorted(shared_modules))

############################  二、PID统计量分析 ##########################
# 1、创建模态缺失模型
# 1.1 深拷贝原模型，stream1对应着我们的RGB流，stream2对应着我们的IR流
model_only_IR = copy.deepcopy(model)
model_only_RGB = copy.deepcopy(model)
# model_all_zero = copy.deepcopy(model)

# 1.2 参数置零
zero_out_specific_modules(model_only_IR, stream1_modules) # 置零stream1_modules中的对应层
zero_out_specific_modules(model_only_RGB, stream2_modules) # 置零stream2_modules中的对应层
# zero_out_model_params(model_all_zero) # 置零所有层

# 检查模型参数是否成功置零
# check_zeroed_params(model_only_IR)
# check_zeroed_params(model_only_RGB)
# check_zeroed_params(model_all_zero)

# 2、预测得到对应的输出

# 循环预测，保存json以及转为word
# eval_all_combination(
#     model=model,
#     model_only_RGB=model_only_RGB,
#     model_only_IR=model_only_IR,
#     data_cfg='./data/multispectral/LLVIP.yaml',
#     device = device
#     )
# json_to_word(json_path='./evaluation_results.json',word_path='./evaluation_results.docx')


# 2.1 无缺失模式
ap_class12, p_y_x12, p_y, dataloader_len = multimodal_eval(
    model=model,           # 三种模型：model,model_only_RGB,model_only_IR
    data_cfg='./data/multispectral/FLIR_aligned.yaml', # M3FD_8_2  LLVIP
    device=device,
    split='train',                        # 可选：'train' / 'val' / 'all'
    zero_mode=None,                      # 可选：None, 'rgb', 'ir'
    batch_size=64
)

# print("p_y_x12:[:20]")
# print(p_y_x12[:20])
# print("p_y:[:20]")
# print(p_y[:20])

# print("p_y_x12:[len(p_y_x12)-3:]")
# print(p_y_x12[len(p_y_x12)-3:])
# print("p_y:[len(p_y)-3:]")
# print(p_y[len(p_y)-3:])

# 2.2 仅使用IR信息，RGB参数置零且缺失RGB输入(可以选择包含所有的IR和RGB结果，并不影响最后的输出)
ap_class2, p_y_x2 ,p_y, dataloader_len = multimodal_eval(
    model=model_only_IR,           # 三种模型：model,model_only_RGB,model_only_IR
    data_cfg='./data/multispectral/FLIR_aligned.yaml',
    device=device,
    split='train',                        # 可选：'train' / 'val' / 'all'
    zero_mode='rgb',                      # 可选：None, 'rgb', 'ir'
    batch_size=64
)

# 2.3 仅使用RGB信息，IR参数置零且缺失IR输入(可以选择包含所有的IR和RGB结果，并不影响最后的输出)
ap_class1, p_y_x1, p_y, dataloader_len = multimodal_eval(
    model=model_only_RGB,           # 三种模型：model,model_only_RGB,model_only_IR
    data_cfg='./data/multispectral/FLIR_aligned.yaml',
    device=device,
    split='train',                        # 可选：'train' / 'val' / 'all'
    zero_mode='ir',                      # 可选：None, 'rgb', 'ir'
    batch_size=64
)

# 3、计算互信息
# 3.1 预处理p_y
p_y =torch.sum(p_y, dim=0) / p_y.size(0)
p_y[p_y == 0] += 1e-8
p_y[p_y == 1] -= 1e-8
print(f"p_y:{p_y}")

# 3.2 计算互信息mi_y_x1
p_y_x1[p_y_x1 == 0] += 1e-8  # 防止对数运算时出现负无穷
p1 = p_y_x1.detach().clone()
log_p_y_x1 = torch.log(p1)
mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (log_p_y_x1 - torch.log(p_y)[None]), dim=-1))
mi_y_x1 = max(mi_y_x1,0) # 有一些计算误差，最小为0
print(f"mi_y_x1:{mi_y_x1}")

# 3.3 计算互信息mi_y_x2
p_y_x2[p_y_x2 == 0] += 1e-8  # 防止对数运算时出现负无穷
p2 = p_y_x2.detach().clone()
log_p_y_x2 = torch.log(p2)
mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (log_p_y_x2 - torch.log(p_y)[None]), dim=-1))
mi_y_x2 = max(mi_y_x2,0) 
print(f"mi_y_x2:{mi_y_x2}")

# 3.4 计算互信息mi_y_x12
p_y_x12[p_y_x12 == 0] += 1e-8  # 防止对数运算时出现负无穷
p12 = p_y_x12.detach().clone()
log_p_y_x12 = torch.log(p12)
mi_y_x12 = torch.mean(torch.sum(p_y_x12 * (log_p_y_x12 - torch.log(p_y)[None]), dim=-1))
mi_y_x12 = max(mi_y_x12,0) 
print(f"mi_y_x12:{mi_y_x12}")

# 4、计算冗余性和唯一性
redundancy = mi_y_x12 - mi_y_x1 - mi_y_x2
unique_RGB = mi_y_x12 - mi_y_x2
unique_IR = mi_y_x12 - mi_y_x1

print(f"redundancy:{redundancy}")
print(f"unique_RGB:{unique_RGB}")
print(f"unique_IR:{unique_IR}")

## 清理不必要的信息
del model_only_IR,model_only_RGB
gc.collect()
torch.cuda.empty_cache()

#############################  三、剪枝设计 #####################
# 1、初始化工作
# 1.1 忽略剪枝的层（比如最后的分类层或输出层）
ignored_layers = []

from models.yolo_test import Detect
for m in model.modules():
    if isinstance(m, Detect):
        ignored_layers.append(m)

print('---------ignored_layers-----------')
print(ignored_layers)
print('---------ignored_layers-----------')

# 1.2 定义重要性度量标准
imp = tp.importance.GroupNormImportance(p=1)

# 1.3 计算剪枝前的模型计算量和参数量
model = model.float() # 需要转为全精度
base_macs, base_nparams = tp.utils.count_ops_and_params(model,example_inputs)
base_macs_G = base_macs / 1e9  # 转换为G
base_nparams_M = base_nparams / 1e6  # 转换为M
print(f"原始模型 MACs: {base_macs_G:.2f}G, 参数量: {base_nparams_M:.2f}M")

# 2、剪枝率分配，采用sofrtmax和差异惩罚来平滑剪枝率
# 2.1 基础剪枝率
base_pruning_ratio = 0.5

# 2.2 参数调节（可以根据实验调优）
multiple_num = 2
gamma = 5                      # softmax 温度系数，控制分配敏感度
lambda_ = 0.15                    # 平衡程度，越接近0越平均，越接近1越依赖信息量

# 2.3 softmax 权重计算（唯一性越高，权重越大）
uniqueness = np.array([unique_RGB, unique_IR])
s = np.exp(gamma * uniqueness)
w = s / np.sum(s)

# 2.4 初步剪枝率分配（信息量越高剪得越少）
total_pruning_ratio = base_pruning_ratio * multiple_num
raw_pruning = total_pruning_ratio * (1 - w)

# 2.5 平滑处理，压缩剪枝率差距（靠近均值）
mean_prune = total_pruning_ratio / 2
smoothed_pruning = lambda_ * raw_pruning + (1 - lambda_) * mean_prune

# 2.6 平滑后的剪枝率，不应该大于0.98
RGB_pruning_ratio = min(smoothed_pruning[0],0.95)
IR_pruning_ratio = min(smoothed_pruning[1],0.95)

# 2.7 打印输出
print(f"RGB pruning ratio: {RGB_pruning_ratio:.4f}")
print(f"IR  pruning ratio: {IR_pruning_ratio:.4f}")

# 3、分配RGB和IR的剪枝率
ratio_dict = dict()

# 3.1 判断当前层是否在某一层，注意要加上'.'，否则想要忽略'model.2','model.2x'也会被忽略
def is_in_stream(name,stream):
    module_prefix = ".".join(name.split(".")[:2])  # 获取model.x
    # print(f"module_prefix:{module_prefix} ")
    return any(module_prefix == prefix for prefix in stream)

# 3.2 循环赋值
def assign_layers_ratio(ratio_dict,stream,model,ratio):
    for name, module in model.named_modules():
        # 整体的model需要skip，这个不用分配
        if(name == 'model' or name == ''):
            print(f"skip:{name}")
            continue
            
        if is_in_stream(name,stream):  # 在 stream 的的进行赋值
            # print(f"name:{name}")
            ratio_dict[module] = ratio

# 3.3 赋值RGB,IR层
assign_layers_ratio(ratio_dict,stream1_modules,model,RGB_pruning_ratio)
assign_layers_ratio(ratio_dict,stream1_modules,model,IR_pruning_ratio)

# 4、剪枝
# 创建 pruner
pruner = tp.pruner.GroupNormPruner(
    model,
    example_inputs,
    global_pruning = True,
    importance = imp,
    pruning_ratio = base_pruning_ratio,
    pruning_ratio_dict = ratio_dict,
    ignored_layers = ignored_layers
)

pruner.step()

macs, nparams = tp.utils.count_ops_and_params(model,example_inputs) 
macs_G = macs / 1e9  # 转换为G
nparams_M = nparams / 1e6  # 转换为M
print(f"全部剪枝完成")
print(f"MACs 剩余: {macs_G:.2f}G 剩余百分比{macs/base_macs:.4f} 剪枝率{1-macs/base_macs:.4f}")
print(f"参数剩余: {nparams_M:.2f}M 剩余百分比{nparams/base_nparams:.4f} 剪枝率{1-nparams/base_nparams:.4f}")

print(f"base_pruning_ratio:{base_pruning_ratio}")
print(f"multiple_num:{multiple_num}")
print(f"gamma:{gamma}")
print(f"lambda:{lambda_}")


print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# 清理缓存
del pruner,ignored_layers,ratio_dict
gc.collect()
torch.cuda.empty_cache()

#############################  四、剪枝后精度恢复 #####################
start_train(model)

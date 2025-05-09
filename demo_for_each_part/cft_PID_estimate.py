import torch
import torch.nn as nn
import copy
from utils.parameters_set_zero import zero_out_model_params, zero_out_specific_modules,analyze_output, check_zeroed_params, check_model_output_is_zero  
from utils.yolo_branch_analyzer import YoloStreamAnalyzer
from utils.model_eval import multimodal_eval,eval_all_combination,json_to_word
import json
from utils.loss import ComputeLoss

############################  一、双分支分析 ##########################
# 1、加载模型和输入
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load("cft_best_llvip.pt", map_location=device)
model = ckpt['model'].float().to(device)
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
    data_cfg='./data/multispectral/LLVIP.yaml',
    device=device,
    split='val',                        # 可选：'train' / 'val' / 'all'
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
    data_cfg='./data/multispectral/LLVIP.yaml',
    device=device,
    split='val',                        # 可选：'train' / 'val' / 'all'
    zero_mode='rgb',                      # 可选：None, 'rgb', 'ir'
    batch_size=64
)

# 2.3 仅使用RGB信息，IR参数置零且缺失IR输入(可以选择包含所有的IR和RGB结果，并不影响最后的输出)
ap_class1, p_y_x1, p_y, dataloader_len = multimodal_eval(
    model=model_only_RGB,           # 三种模型：model,model_only_RGB,model_only_IR
    data_cfg='./data/multispectral/LLVIP.yaml',
    device=device,
    split='val',                        # 可选：'train' / 'val' / 'all'
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

# ############################  三、剪枝设计 #####################

# 这部分都在pruning_with_PID里面



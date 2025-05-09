################################### 针对Yolo结构的识别 #################################
# v5：添加了一些辅助输出，名曲了不需要
# 问题：在v4的基础上，完善对于LayerList的构造，目前存在的问题是有一些依赖没有被识别到，我们需要改进LayerList的处理流程
# 麻烦：原来的dependency graph虽然有记录add操作的依赖，但是没法映射回原来的model的对应的层，是作为ElementWiseOpt
# 有无必要性：没有，Add这种操作本身是没有参数的，我们能够正确识别到Stream1和Stream2就够了。

################################### 链表结构 #################################
# 具体操作为使用链表的形式构造layer_pairs，
# 每个节点需要存储：
# 1、指向该节点的节点组；
# 2、该节点指向的节点组；
# 3、该节点是否属于_ElementWiseOp_；

# 链表需要提供的方法：
# 1、给定一个节点，获取其入度；
# 2、给定一个节点，获取其出度；
# 3、删除节点：如果一个节点的入度为1和出度均为1，则可以删除该节点，否则该节点不能删除；
# 4、从指定节点开始遍历链表，注意需要处理节点出度>1的情况，此时按照深度优先遍历即可；
# 5、此外链表需要存储其中的所有nodes


import torch
import torch.nn as nn
import torch_pruning as tp
import networkx as nx
import matplotlib.pyplot as plt
import re

########################### 1、模型载入 ############################
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 本地预训练模型
weights_path = "cft_best_llvip.pt"
ckpt = torch.load(weights_path, map_location=device)  # 加载 checkpoint
model = ckpt['model']  # 提取 PyTorch 模型
model = model.float()  # 处理模型

torch.manual_seed(0)  # 固定随机种子
img_rgb = torch.randn(8, 3, 640, 640).to(device)  # RGB 图像输入
img_ir = torch.randn(8, 3, 640, 640).to(device)   # 红外图像输入
example_inputs=[img_rgb, img_ir]

for m in model.modules():
    m.requires_grad_(True)

####################### 2、创建依赖图+构造链表来实现图结构 ###############################
# 2.1创建依赖图
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs)

# print("Modules in DependencyGraph:")
# for i, module in enumerate(DG.module2node.keys()):
#     print(f"{i}: {module}")

# with open("cft_dependency_graph_modules.txt", "w") as f:
#     for i, module in enumerate(reversed(DG.module2node.keys())):
#         print(f"{i}: {module}", file=f)

# 2.2 用于从DG中构造一个便于遍历的图结构的链表

# 获取节点标签，例如：
# Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))变为Conv2d
# _ElementWiseOp_2(AddBackward0)变为AddBackward0

def get_node_label(module, layer):
    module_str = str(module)
    if "ElementWiseOp" in module_str:
        # 提取括号内的内容
        match = re.search(r'\((.*?)\)', module_str)
        module_type = match.group(1) if match else "Unknown"
    else:
        # 提取括号外的类型名称
        module_type = re.sub(r'\(.*\)', '', module_str).split('.')[-1]
    return f"{layer}:{module_type}"

# 链表节点，用于构造图
class LayerNode:
    def __init__(self, module, layer, path=""):
        self.module = module   # 自身的module
        self.layer = layer     # 在DG中是几层
        self.path = path       # 在原模型中的位置，例如model.0.conv这种
        self.in_nodes = set()  # 指向该节点的节点组
        self.out_nodes = set() # 该节点指向的节点组
        self.is_elementwise = "ElementWiseOp" in str(module) # 是否是ElementWiseOp，一般是Add、Relu、Concat这些操作
    
    def __str__(self):
        if(self.path != '<unknown>'):
            return f"{self.path}"
        else:
            return f"{self.module}"

    def __hash__(self):
        return hash(id(self))  # 或者 hash(self.module)

    def __eq__(self, other):
        return isinstance(other, LayerNode) and self.module == other.module

# 链表实现
class LayerLinkedList:
    # 维护所有的nodes
    def __init__(self):
        self.nodes = {}
        self.module_path_map = {}  # 保存 module -> path

    # 维护每个module->paht
    def build_path_map(self, model, prefix="model"):
        for name, module in model.named_modules():
            path = f"{prefix}.{name}" if name else prefix
            self.module_path_map[module] = path

    # 增加一条边
    def add_edge(self, source_module, target_module, source_layer, target_layer):
        if source_module not in self.nodes:
            source_path = self.module_path_map.get(source_module, "<unknown>")
            self.nodes[source_module] = LayerNode(source_module, source_layer, source_path)

        if target_module not in self.nodes:
            target_path = self.module_path_map.get(target_module, "<unknown>")
            self.nodes[target_module] = LayerNode(target_module, target_layer, target_path)

        self.nodes[source_module].out_nodes.add(self.nodes[target_module])
        self.nodes[target_module].in_nodes.add(self.nodes[source_module])
    
    # 获取指定节点的入度
    def get_in_degree(self, module):
        return len(self.nodes[module].in_nodes) if module in self.nodes else 0
    
    # 获取指定节点的出度
    def get_out_degree(self, module):
        return len(self.nodes[module].out_nodes) if module in self.nodes else 0

    # 只有当节点的入度和出度均为1的时候才能够移除，否则会导致依赖关系错误
    def can_remove_node(self, module):
        if module not in self.nodes:
            return False
        node = self.nodes[module]
        return len(node.in_nodes) == 1 and len(node.out_nodes) == 1
    

    # 移除节点
    def remove_node(self, module):
        if not self.can_remove_node(module):
           return False
        node = self.nodes[module]
        parent = next(iter(node.in_nodes))
        child = next(iter(node.out_nodes))
        parent.out_nodes.remove(node)
        child.in_nodes.remove(node)
        parent.out_nodes.add(child)
        child.in_nodes.add(parent)
        del self.nodes[module]
        return True
    
    # 从给定的结点开始遍历链表
    def traverse_from(self, module, visited=None):
        if visited is None:
            visited = set()
        
        if module not in self.nodes or module in visited:
            return
        
        visited.add(module)
        node = self.nodes[module]
        for next_node in node.out_nodes:
            self.traverse_from(next_node.module, visited)

# ######################## 3、处理得到各个分支与共享节点 ####################### 

# 3.1 使用链表存储 layer_pairs
layer_index = {module: len(DG.module2node) - i - 1 for i, (module, node) in enumerate(DG.module2node.items())}
layer_list = LayerLinkedList()
layer_list.build_path_map(model) # 构建module -> path

# 遍历 DG，生成 (层数, 依赖关系) 的 pair
for module, node in DG.module2node.items():
    module_layer = layer_index[module]
    for dep in node.dependencies:
        source_layer = layer_index[dep.source.module]
        target_layer = layer_index[dep.target.module]
        if source_layer < target_layer:  # 只添加编号小的指向编号大的边
            layer_list.add_edge(dep.source.module, dep.target.module, source_layer, target_layer)

# print("------ LayerList: All Nodes Sorted by Path ------")
# for i, node in enumerate(sorted(layer_list.nodes.values(), key=lambda n: n.path)):
#     print(f"{i}: {node.path} ({node.layer}) - {type(node.module).__name__}")


# print("------------------- Node In-Degree and Out-Degree--------------------")
# for module, node in layer_list.nodes.items():
#     in_degree = len(node.in_nodes)
#     out_degree = len(node.out_nodes)
#     print(f"Node: {node}, In-Degree: {in_degree}, Out-Degree: {out_degree}")


# 3.2 获取两个 Focus 层对应的第一个conv（stream1 和 stream2 的起点）
def find_focus_convs(model):
    result = {}
    count = 0

    for name, module in model.named_modules():
        if name.endswith('.conv.conv') and isinstance(module, nn.Conv2d):
            # 检查上一级模块是否为 Focus（防止误抓）
            parent_name = ".".join(name.split(".")[:-1])
            parent = dict(model.named_modules()).get(parent_name, None)
            if parent and parent.__class__.__name__ == 'Conv':
                grandparent_name = ".".join(name.split(".")[:-2])
                grandparent = dict(model.named_modules()).get(grandparent_name, None)
                if grandparent and grandparent.__class__.__name__ == 'Focus':
                    count += 1
                    stream_name = f'stream{count}_head'
                    result[stream_name] = (name, module)

    return result


# 找到对应的conv并输出结果
focus_convs = find_focus_convs(model)
print("-------------- Focus模块中第一个conv --------------------")
for k, (path, layer) in focus_convs.items():
    print(f"{k}: {path} {layer}")

# 从链表中找到对应的 LayerNode
stream1_head_node = None
stream2_head_node = None

#  print("-------------- 对应的LayerList中的Node --------------------")
for k, (path, module) in focus_convs.items():
    #  print(f"{k}: {path} {module}")
    node = layer_list.nodes.get(module, None)
    if node is not None:
        if k == "stream1_head":
            stream1_head_node = node
        elif k == "stream2_head":
            stream2_head_node = node

if stream1_head_node is None or stream2_head_node is None:
    raise RuntimeError("未能在链表中找到 stream1/stream2 对应的节点。请检查构图是否正确。")

# 3.3 获取 stream1 和 stream2
stream1, stream2, shared_nodes = set(), set(), set()
enable_elementwise_removal=True # 需要删除对应的ElemWiseOpt节点

# 使用栈来遍历链表中的元素
def collect_stream(node, stream_set,enable_elementwise_removal):
    visited = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in visited or current.module not in layer_list.nodes:
            continue
        visited.add(current)
        
        # 如果是 ElementWiseOp 类型，并且仍在 layer_list 中，同时可以被删除(出度和入度均为1),尝试删除
        should_delete = enable_elementwise_removal and current.is_elementwise and layer_list.can_remove_node(current.module)

        if not should_delete:
            stream_set.add(current)
        
        if should_delete:
            layer_list.remove_node(current.module)

        stack.extend(current.out_nodes)

collect_stream(stream1_head_node, stream1,enable_elementwise_removal=enable_elementwise_removal)
collect_stream(stream2_head_node, stream2, enable_elementwise_removal=enable_elementwise_removal)

# 3.4 计算 shared_nodes，更新stream1、stream并且排序
shared_nodes = stream1 & stream2
stream1 -= shared_nodes
stream2 -= shared_nodes

stream1 = sorted(stream1, key=lambda n: n.layer)
stream2 = sorted(stream2, key=lambda n: n.layer)
shared_nodes = sorted(shared_nodes, key=lambda n: n.layer)

# 3.5 输出结果
print("----------------stream1------------------")
for i,node in enumerate(stream1):
    print(f"stream1:{i} {node}")

print("----------------stream2------------------")
for i,node in enumerate(stream2):
    print(f"stream2:{i} {node}")

print("----------------shared_nodes------------------")
for i,node in enumerate(shared_nodes):
    print(f"shared_nodes:{i} {node}")

######################## 4、绘制依赖图 ########################### 
# 4.1 生成有向图 G
G = nx.DiGraph()

# 只取 stream1、stream2 和 shared_nodes 的节点和边，不影响原始数据
nodes_to_draw = set(stream1) | set(stream2) | set(shared_nodes)

for node in nodes_to_draw:
    for out_node in node.out_nodes:
        if out_node in nodes_to_draw:
            color = 'blue' if node in stream1 else 'red' if node in stream2 else 'green'
            G.add_edge(node, out_node, color=color)

# 4.2 计算节点布局
x_offset = 3.0
y_spacing = 0.8
text_offset = 0.2  # 让文本更贴近节点
pos = {}

for i, node in enumerate(sorted(stream1, key=lambda n: n.layer)):
    pos[node] = (-x_offset, -i * y_spacing)

for i, node in enumerate(sorted(stream2, key=lambda n: n.layer)):
    pos[node] = (x_offset, -i * y_spacing)

if shared_nodes:
    first_shared_node = next(iter(shared_nodes))
    pos[first_shared_node] = (0, min(pos.values(), key=lambda x: x[1])[1] - y_spacing)

for i, node in enumerate(sorted(shared_nodes, key=lambda n: n.layer)):
    pos[node] = (0, -i * y_spacing)

# 计算 ylim 和 xlim
min_y = min(pos.values(), key=lambda x: x[1])[1] - 1
max_y = max(pos.values(), key=lambda x: x[1])[1] + 1
min_x = -x_offset - 1.0
max_x = x_offset + 1.0

# 4.3 绘制依赖图 x轴和y轴宽度
plt.figure(figsize=(10, 50))

edges = G.edges()
edge_colors = [G[u][v]['color'] for u, v in edges]

nx.draw(G, pos, with_labels=False, node_size=3000, node_color='lightblue',
        font_size=16, edge_color=edge_colors, arrows=True)

# 调整文字，使其在节点下方
for node, (x, y) in pos.items():
    plt.text(x, y - text_offset, str(node), fontsize=12, ha='center', fontweight='bold')

# 添加 stream1, shared, stream2 标注
plt.text(-x_offset, max_y - text_offset, "Stream 1", fontsize=16, ha='center', fontweight='bold', color='blue')
plt.text(0, max_y - text_offset, "Shared Nodes", fontsize=16, ha='center', fontweight='bold', color='green')
plt.text(x_offset, max_y - text_offset, "Stream 2", fontsize=16, ha='center', fontweight='bold', color='red')

plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.savefig("cft_dependency_graph_v5.png", bbox_inches='tight')

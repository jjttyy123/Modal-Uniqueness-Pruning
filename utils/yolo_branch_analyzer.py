import torch
import torch.nn as nn
import torch_pruning as tp
import re
import networkx as nx
import matplotlib.pyplot as plt

class YoloStreamAnalyzer:
    """
    该类用于分析YOLO模型中的双分支结构，提取stream1、stream2以及共享节点(shared_nodes)。
    """

    def __init__(self, model, example_inputs, enable_elementwise_removal=True):
        """
        初始化分析器，构建依赖图和链表结构。
        :param model: 传入的YOLO模型
        :param example_inputs: 模型的示例输入
        :param enable_elementwise_removal: 是否启用删除ElementWiseOp节点的功能
        """
        self.model = model
        self.example_inputs = example_inputs
        self.enable_elementwise_removal = enable_elementwise_removal

        # 构建依赖图
        self.DG = tp.DependencyGraph()
        self.DG.build_dependency(model, example_inputs)

        # 创建层索引，用于按层级处理
        self.layer_index = {module: len(self.DG.module2node) - i - 1 
                            for i, (module, node) in enumerate(self.DG.module2node.items())}

        # 构建层链表
        self.layer_list = self._build_layer_linked_list()

    class LayerNode:
        """
        每个LayerNode表示一个网络层的节点，存储了该层的模块、路径、入节点和出节点等信息。
        """
        def __init__(self, module, layer, path=""):
            """
            :param module: 模块本身
            :param layer: 层的编号
            :param path: 在模型中的路径
            """
            self.module = module
            self.layer = layer
            self.path = path
            self.in_nodes = set()  # 入节点集合
            self.out_nodes = set()  # 出节点集合
            self.is_elementwise = "ElementWiseOp" in str(module)  # 判断是否是ElementWiseOp类型的节点（如Add等）

        def __str__(self):
            """返回节点的路径信息或模块信息"""
            return self.path if self.path != '<unknown>' else str(self.module)

        def __hash__(self):
            """为节点提供唯一标识"""
            return hash(id(self))

        def __eq__(self, other):
            """判断两个节点是否相等"""
            return isinstance(other, type(self)) and self.module == other.module

    class LayerLinkedList:
        """
        用于管理模型中各层之间依赖关系的链表结构。
        """
        def __init__(self, model):
            """
            初始化链表，创建模块路径映射表。
            :param model: PyTorch模型
            """
            self.nodes = {}  # 保存节点
            self.module_path_map = {}  # 保存模块到路径的映射
            self.build_path_map(model)

        def build_path_map(self, model, prefix="model"):
            """构建模型的路径映射表"""
            for name, module in model.named_modules():
                path = f"{prefix}.{name}" if name else prefix
                self.module_path_map[module] = path

        def add_edge(self, source_module, target_module, source_layer, target_layer):
            """
            向链表中添加依赖关系（即边）。
            :param source_module: 源模块
            :param target_module: 目标模块
            :param source_layer: 源层的编号
            :param target_layer: 目标层的编号
            """
            if source_module not in self.nodes:
                source_path = self.module_path_map.get(source_module, "<unknown>")
                self.nodes[source_module] = YoloStreamAnalyzer.LayerNode(source_module, source_layer, source_path)

            if target_module not in self.nodes:
                target_path = self.module_path_map.get(target_module, "<unknown>")
                self.nodes[target_module] = YoloStreamAnalyzer.LayerNode(target_module, target_layer, target_path)

            # 建立出入节点关系
            self.nodes[source_module].out_nodes.add(self.nodes[target_module])
            self.nodes[target_module].in_nodes.add(self.nodes[source_module])

        def can_remove_node(self, module):
            """检查一个节点是否可以删除（入度和出度均为1）"""
            node = self.nodes.get(module)
            return node and len(node.in_nodes) == 1 and len(node.out_nodes) == 1

        def remove_node(self, module):
            """删除一个节点，确保不会破坏依赖关系"""
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

    def _build_layer_linked_list(self):
        """
        根据依赖图构建层链表（LayerLinkedList）。
        """
        layer_list = self.LayerLinkedList(self.model)
        for module, node in self.DG.module2node.items():
            module_layer = self.layer_index[module]
            for dep in node.dependencies:
                source_layer = self.layer_index[dep.source.module]
                target_layer = self.layer_index[dep.target.module]
                if source_layer < target_layer:
                    layer_list.add_edge(dep.source.module, dep.target.module, source_layer, target_layer)
        return layer_list

    def find_focus_convs(self):
        """
        查找模型中的Focus模块及其后续的Conv层，作为stream1和stream2的起始节点。
        :return: 返回包含stream1_head和stream2_head的字典
        """
        result, count = {}, 0
        for name, module in self.model.named_modules():
            if name.endswith('.conv.conv') and isinstance(module, nn.Conv2d):
                parent_name = ".".join(name.split(".")[:-1])
                parent = dict(self.model.named_modules()).get(parent_name)
                grandparent_name = ".".join(name.split(".")[:-2])
                grandparent = dict(self.model.named_modules()).get(grandparent_name)
                if parent and grandparent and parent.__class__.__name__ == 'Conv' and grandparent.__class__.__name__ == 'Focus':
                    count += 1
                    result[f'stream{count}_head'] = module
        return result

    def collect_stream(self, node, stream_set):
        """
        从指定节点开始收集stream中的所有节点。
        :param node: 起始节点
        :param stream_set: 存储stream的集合
        """
        visited, stack = set(), [node]
        while stack:
            current = stack.pop()
            if current in visited or current.module not in self.layer_list.nodes:
                continue
            visited.add(current)

            # 如果是ElementWiseOp节点并且可以删除，尝试删除
            if not (self.enable_elementwise_removal and current.is_elementwise and self.layer_list.can_remove_node(current.module)):
                stream_set.add(current)
            else:
                self.layer_list.remove_node(current.module)

            stack.extend(current.out_nodes)

    def analyze(self):
        """
        分析模型，提取stream1、stream2和shared_nodes。
        :return: 返回sorted的stream1、stream2和shared_nodes
        """
        focus_convs = self.find_focus_convs()
        stream1_head = self.layer_list.nodes.get(focus_convs.get('stream1_head'))
        stream2_head = self.layer_list.nodes.get(focus_convs.get('stream2_head'))

        if not stream1_head or not stream2_head:
            raise RuntimeError("无法找到stream1/stream2起始节点")

        stream1, stream2 = set(), set()
        self.collect_stream(stream1_head, stream1)
        self.collect_stream(stream2_head, stream2)

        shared_nodes = stream1 & stream2
        stream1 -= shared_nodes
        stream2 -= shared_nodes

        return (
            sorted(stream1, key=lambda n: n.layer),
            sorted(stream2, key=lambda n: n.layer),
            sorted(shared_nodes, key=lambda n: n.layer)
        )

    def plot_dependency_graph(self, stream1, stream2, shared_nodes, filename="dependency_graph.png"):
        """
        绘制模型的依赖图，并保存为文件。
        :param stream1: 第一个流（分支）
        :param stream2: 第二个流（分支）
        :param shared_nodes: 共享的节点
        :param filename: 保存图像的文件名
        """
        G = nx.DiGraph()

        # 只取 stream1、stream2 和 shared_nodes 的节点和边，不影响原始数据
        nodes_to_draw = set(stream1) | set(stream2) | set(shared_nodes)

        # 创建图的边
        for node in nodes_to_draw:
            for out_node in node.out_nodes:
                if out_node in nodes_to_draw:
                    color = 'blue' if node in stream1 else 'red' if node in stream2 else 'green'
                    G.add_edge(node, out_node, color=color)

        # 计算节点布局
        x_offset = 3.0
        y_spacing = 0.8
        text_offset = 0.2  # 让文本更贴近节点
        pos = {}

        # 为stream1、stream2和共享节点分配位置
        for i, node in enumerate(sorted(stream1, key=lambda n: n.layer)):
            pos[node] = (-x_offset, -i * y_spacing)

        for i, node in enumerate(sorted(stream2, key=lambda n: n.layer)):
            pos[node] = (x_offset, -i * y_spacing)

        if shared_nodes:
            first_shared_node = next(iter(shared_nodes))
            pos[first_shared_node] = (0, min(pos.values(), key=lambda x: x[1])[1] - y_spacing)

        for i, node in enumerate(sorted(shared_nodes, key=lambda n: n.layer)):
            pos[node] = (0, -i * y_spacing)

        # 计算 y 轴和 x 轴的范围
        min_y = min(pos.values(), key=lambda x: x[1])[1] - 1
        max_y = max(pos.values(), key=lambda x: x[1])[1] + 1
        min_x = -x_offset - 1.0
        max_x = x_offset + 1.0

        # 绘制依赖图
        plt.figure(figsize=(10, 50))

        edges = G.edges()
        edge_colors = [G[u][v]['color'] for u, v in edges]

        # 绘制图形并标注节点
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
        plt.savefig(filename, bbox_inches='tight')

# utils/yolo_model_zero.py

import torch
import torch.nn as nn

def zero_out_model_params(model):
    """
    将模型中的所有层的权重和偏置置为0，包括卷积层、全连接层和BatchNorm层。
    :param model: PyTorch模型对象
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 0.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)  # 防止除以0
            m.eval()  # BN在eval模式下不使用batch statistics

def zero_out_specific_modules(model, modules_to_zero):
    """
    置零指定模块的权重。
    :param model: PyTorch模型对象
    :param modules_to_zero: 包含需要置零的模块路径的集合
    """
    for name, m in model.named_modules():  # 使用named_modules()来获取模块的路径和模块对象
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            # 拆分路径，获取模型的第二部分（例如从model.16.m.5变成model.16）
            module_prefix = ".".join(name.split(".")[:2])  # 获取model.x
            # 仅在模块路径完全匹配或路径以指定路径为开头时执行置零
            if module_prefix in modules_to_zero:
                # 置零权重
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 0.0)
                # 置零偏置
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                # 置零BatchNorm的running mean和variance
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.running_mean.zero_()
                    m.running_var.fill_(1.0)  # 防止除以0
                    m.eval()

def analyze_output(output, name="output", atol=1e-6):
    """
    分析模型输出是否接近全零。
    :param output: 输出（Tensor、列表或元组）
    :param name: 输出名称（用于打印信息）
    :param atol: 绝对容差，用于判断是否接近全零
    """
    if isinstance(output, torch.Tensor):
        if output.numel() == 0:
            print(f" [{name}] 是空Tensor，没有检测到任何目标。")
        elif torch.allclose(output, torch.zeros_like(output), atol=atol):
            print(f" [{name}] 近似全0")
        else:
            print(f" [{name}] 非全0 -> mean: {output.mean():.6f}, max: {output.max():.6f}, min: {output.min():.6f}")
    elif isinstance(output, (list, tuple)):
        if len(output) == 0:
            print(f" [{name}] 是空列表/元组")
        else:
            for idx, item in enumerate(output):
                analyze_output(item, name=f"{name}[{idx}]")
    else:
        print(f" [{name}] 类型 {type(output)} 不支持分析")

def check_zeroed_params(model):
    """
    检查模型中的每一层的权重和偏置是否已经置零。
    :param model: PyTorch模型对象
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            # 检查权重
            if hasattr(m, 'weight') and m.weight is not None:
                if torch.allclose(m.weight, torch.zeros_like(m.weight)):
                    print(f"{name}: weight is zeroed.")
                else:
                    print(f"{name}: weight is NOT zeroed.")
            # 检查偏置
            if hasattr(m, 'bias') and m.bias is not None:
                if torch.allclose(m.bias, torch.zeros_like(m.bias)):
                    print(f"{name}: bias is zeroed.")
                else:
                    print(f"{name}: bias is NOT zeroed.")

def check_model_output_is_zero(model, img_rgb, img_ir):
    """
    检查模型的输出是否接近零。
    :param model: PyTorch模型对象
    :param img_rgb: RGB图像输入
    :param img_ir: 红外图像输入
    """
    with torch.no_grad():
        out, train_out = model(img_rgb, img_ir)
        # 检查输出是否接近零
        analyze_output(out, name="out")  # 使用analyze_output方法，检查输出是否全零
        analyze_output(train_out, name="train_out")  # 检查train_out（P3, P4, P5等特征图）

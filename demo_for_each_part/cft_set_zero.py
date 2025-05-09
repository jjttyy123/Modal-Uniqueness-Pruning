import torch
import torch.nn as nn

# ===== 1. 加载模型 =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = "cft_best_llvip.pt"
ckpt = torch.load(weights_path, map_location=device)
model = ckpt['model']
model = model.float().to(device)
model.eval()

# ===== 2. 将所有权重和偏置置为0，包括 BN =====
def zero_out_model_params(model):
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

zero_out_model_params(model)

# ===== 3. 构造全零输入 =====
img_rgb = torch.zeros(8, 3, 640, 640).to(device)
img_ir = torch.zeros(8, 3, 640, 640).to(device)

# ===== 4. 前向传播，查看输出是否全为0 =====
with torch.no_grad():
    out, train_out = model(img_rgb, img_ir)

print("=== 原始模型输出结构 ===")
print("train_out:", type(train_out))
print("out:", type(out))
print("train_out 长度:", len(train_out))
print("out 长度:", len(out))


# ===== 5. 分析输出是否为全0 =====
def analyze_output(output, name="output", atol=1e-6):
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


# 分析train_out (P3, P4, P5等特征图)
print("\n train_out 各层输出检查：")
analyze_output(train_out, name="train_out")

# 分析out (8张图像的预测结果)
print("\n out 输出检查（后处理后，每张图片预测结果）：")
analyze_output(out, name="out")
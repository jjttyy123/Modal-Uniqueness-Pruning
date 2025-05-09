import torch

# 构造一个简单的测试案例
p_y_x1 = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.5, 0.7]], dtype=torch.float32)
p_y = torch.tensor([0.4, 0.6], dtype=torch.float32)

# 防止对数运算时出现负无穷
p_y_x1[p_y_x1 == 0] += 1e-8
p_y[p_y == 0] += 1e-8  # 也确保 p_y 没有零值

# 计算互信息
p1 = p_y_x1.detach().clone()
log_p_y_x1 = torch.log(p1)
mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (log_p_y_x1 - torch.log(p_y)[None]), dim=-1))

# 理论值计算
theoretical_mi = 0.0
for i in range(p_y_x1.shape[0]):
    for j in range(p_y_x1.shape[1]):
        theoretical_mi += p_y_x1[i, j] * (torch.log(p_y_x1[i, j]) - torch.log(p_y[j]))

print(f"Code output: {mi_y_x1.item()}")
print(f"Theoretical value: {theoretical_mi.item()}")
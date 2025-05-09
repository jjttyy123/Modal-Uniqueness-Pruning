from models.experimental import attempt_load # 自带的加载函数，性能表现更好？
import torch
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('runs/train/exp/weights/best.pt', map_location=device)

for key in ckpt:
    print(f'{key}: {type(ckpt[key])} {ckpt[key]}')

save_ckpt = {
    'model': ckpt['model'].zero_grad(),
    'ema': ckpt['ema']
}

torch.save(save_ckpt,'best_no_grad.pt')
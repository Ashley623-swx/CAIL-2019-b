import numpy as np
import torch
from torch.optim import lr_scheduler

# 选择学习率调整策略 1.（默认）余弦模拟退火 2.余弦模拟退火热重启
def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    elif schedule == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif schedule is None:
        return None
    return scheduler


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, num_epochs, iter_per_epoch, cosine_decay=True, excluded_group=None):
        self.excluded_group = excluded_group
        self.base_lrs = {param_group["name"]: param_group["lr"] for param_group in optimizer.param_groups}
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0
        self.cosine_decay = cosine_decay

        self.init_lr()  # so that at first step we have the correct step size

    def get_lr(self, base_lr):
        if self.iter < self.warmup_iter:
            return base_lr * self.iter / self.warmup_iter
        elif not self.cosine_decay:
            return base_lr
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return 0.5 * base_lr * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))
    
    def update_param_groups(self):
        for param_group in self.optimizer.param_groups:
            if not self.excluded_group or not param_group["name"].startswith(self.excluded_group):
                param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])

    def step(self):
        self.update_param_groups()
        self.iter += 1
    
    def init_lr(self):
        self.update_param_groups()
                

class WarmupCosineMomentumScheduler:
    def __init__(self, base_momentum, warmup_epochs, num_epochs, iter_per_epoch, cosine_decay=True):
        self.base_val = 1. - base_momentum
        self.warmup_iter = warmup_epochs * iter_per_epoch
        self.total_iter = num_epochs * iter_per_epoch
        self.current_lr = 0
        self.cosine_decay = cosine_decay

    def get_lr(self, step):
        if step < self.warmup_iter:
            return 1. - self.base_val * step / self.warmup_iter
        elif not self.cosine_decay:
            return 1. - self.base_val
        else:
            decay_iter = self.total_iter - self.warmup_iter
            return 1. - 0.5 * self.base_val * (1 + np.cos(np.pi * (step - self.warmup_iter) / decay_iter))


class CosineSeqLenScheduler:
    def __init__(self, min_val, max_val, num_epochs, iter_per_epoch):
        self.base_val = max_val - min_val
        self.min_val = min_val
        self.max_val = max_val
        self.total_iter = num_epochs * iter_per_epoch

    def get_lr(self, step):
        return self.max_val - 0.5 * self.base_val * (1 + np.cos(np.pi * step / self.total_iter))


class CosineScheduler:
    def __init__(self, min_val, max_val, num_epochs, iter_per_epoch):
        self.min_val = min_val
        self.max_val = max_val
        self.total_iter = num_epochs * iter_per_epoch
        self.current_lr = 0

    def get_lr(self, step):
        if step > self.total_iter:
            return self.min_val
        return 0.5 * (self.max_val - self.min_val) * (1 + np.cos(np.pi * step / self.total_iter)) + self.min_val

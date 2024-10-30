import numpy as np
                

class LinearThenFlatScheduler:
    def __init__(self, initial, final, linear_epochs, iter_per_epoch):
        self.initial = initial
        self.final = final
        self.linear_iters = linear_epochs * iter_per_epoch

    def get_lr(self, step):
        if step < self.linear_iters:
            return self.initial + (self.final - self.initial) * step / self.linear_iters
        return self.final

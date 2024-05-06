import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from monai.transforms import (
    Compose,
    RandFlip,
    RandScaleIntensity,
    RandShiftIntensity,
    NormalizeIntensity,
    ToTensor
)

def torch_acc(y_pred, y_true):
    train_acc = (torch.argmax(y_pred, dim=1) == y_true).float().mean()
    return train_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WarmupExpSchedule(LambdaLR):
    """ Linear warmup and then exponential decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    """
    def __init__(self, optimizer, warmup_steps, t_total, decay_rate=0.95, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.decay_rate = decay_rate
        super(WarmupExpSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps)#/ float(max(1, self.t_total - self.warmup_steps))
        return max(0.0,  (self.decay_rate)**progress)
    
train_transform = Compose(
    [
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandFlip(spatial_axis=2, prob=0.5),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        #RandScaleIntensity(factors=0.1, prob=0.5),
        #RandShiftIntensity(offsets=0.1, prob=0.5),
        ToTensor()
    ]
)

test_transform = Compose(
    [
        NormalizeIntensity(nonzero=True, channel_wise=True),
        ToTensor()
    ]
)
import math
from functools import partial


def get_lr_scheduler(lr_decay_type, lr, min_lr, epochs, warmup_epochs_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_epochs_ratio=0.05, step_num=10):
    def warm_cos_lr(lr, min_lr, epochs, warmup_total_epochs, warmup_lr_start, no_aug_epochs, epoch):
        if epoch <= warmup_total_epochs:
            lr = (lr - warmup_lr_start) * pow(epoch / float(warmup_total_epochs), 2
                                              ) + warmup_lr_start
        elif epoch >= epochs - no_aug_epochs:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (epoch - warmup_total_epochs)
                / (epochs - warmup_total_epochs - no_aug_epochs)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, epoch):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = epoch // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_epochs = min(max(warmup_epochs_ratio * epochs, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_epochs = min(max(no_aug_epochs_ratio * epochs, 1), 15)
        func = partial(warm_cos_lr, lr, min_lr, epochs, warmup_total_epochs, warmup_lr_start, no_aug_epochs)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = epochs / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

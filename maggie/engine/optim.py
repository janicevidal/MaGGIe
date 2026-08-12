import itertools
import torch

import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class WarmupDecayMultiCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup + constant holding + multi‑stage cosine decay with peak scaling.

    Args:
        optimizer: wrapped optimizer.
        max_iters: total number of training iterations.
        warmup_iters: number of warmup iterations (linear increase).
        warmup_factor: initial learning rate factor (e.g., 0.001 => start_lr = base_lr * 0.001).
        delay_iters: iteration at which the first decay stage starts (must be >= warmup_iters).
        decay_steps: list of absolute iteration milestones marking the end of each cosine decay stage.
                     The last value (if less than max_iters) defines the end of the final stage;
                     if the last value >= max_iters, it is truncated to max_iters.
        peak_scale: factor by which the peak LR is multiplied after each stage (soft restart).
        eta_min: minimum LR ratio relative to the current peak (e.g., 0.05).
        last_epoch: initial epoch (iteration) index.
    """
    def __init__(self, optimizer, max_iters, warmup_iters=0, warmup_factor=0.001,
                 delay_iters=0, decay_steps=[], peak_scale=0.8, eta_min=0.05,
                 last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.delay_iters = delay_iters
        self.decay_steps = sorted([s for s in decay_steps if s > delay_iters and s <= max_iters])
        if not self.decay_steps:
            self.decay_steps.append(max_iters)
        self.peak_scale = peak_scale
        self.eta_min = eta_min
        # Ensure delay_iters >= warmup_iters
        assert delay_iters >= warmup_iters, "delay_iters must be >= warmup_iters"
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        base_lrs = self.base_lrs

        # 1) Warmup
        if it < self.warmup_iters:
            # linear warmup
            alpha = it / self.warmup_iters
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in base_lrs]

        # 2) Hold
        if it <= self.delay_iters:
            return [base_lr for base_lr in base_lrs]

        # 3) Multi-stage cosine decay with peak scaling
        # Determine which stage we are in and compute the current peak LR.
        # The first stage starts at delay_iters.
        prev = self.delay_iters
        current_peaks = list(base_lrs)
        for milestone in self.decay_steps:
            if it <= milestone:
                # current stage: from prev to milestone
                length = milestone - prev
                pos = it - prev
                cos_val = (1 + math.cos(math.pi * pos / length)) / 2
                lrs = []
                for peak in current_peaks:
                    lr = peak * cos_val + self.eta_min * (1 - cos_val)
                    lrs.append(lr)
                return lrs
            else:
                prev = milestone
                current_peaks = [peak * self.peak_scale for peak in current_peaks]

        return [self.eta_min for _ in base_lrs]

def build_optim_lr_scheduler(cfg, model):
    def gradient_clipping(optim):
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = 0.01
        enable = False

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    # Define optimizer
    optim_config = cfg.train.optimizer
    if optim_config.name == 'sgd':
        optimizer = gradient_clipping(torch.optim.SGD)(model.parameters(), lr=optim_config.lr, momentum=optim_config.momentum, weight_decay=optim_config.weight_decay)
    elif optim_config.name == 'adam':
        optimizer = gradient_clipping(torch.optim.Adam)(model.parameters(), lr=optim_config.lr, betas=optim_config.betas, weight_decay=optim_config.weight_decay)
    elif optim_config.name == 'adamw':
        optimizer = gradient_clipping(torch.optim.AdamW)(model.parameters(), lr=optim_config.lr, betas=optim_config.betas, weight_decay=optim_config.weight_decay)
    else:
        raise NotImplementedError

    # Define lr scheduler
    scheduler_config = cfg.train.scheduler
    if scheduler_config.name == 'poly':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (cfg.train.max_iter + 1)) ** scheduler_config.power)
    elif scheduler_config.name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.step_size, gamma=scheduler_config.gamma)
    elif scheduler_config.name == 'warmup_decay':
        def lr_lambda(iter):
            if iter < scheduler_config.warmup_iters:
                return iter * 1.0 / scheduler_config.warmup_iters
            else:
                return math.sqrt(scheduler_config.warmup_iters * 1.0 / iter)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_config.name == 'cosine':
        pct_start = scheduler_config.warmup_iters * 1.0 / cfg.train.max_iter
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optim_config.lr, total_steps=cfg.train.max_iter, pct_start=pct_start, anneal_strategy='cos', cycle_momentum=False)
    elif scheduler_config.name == 'warmup_decay_multi':
        scheduler = WarmupDecayMultiCosineLR(
            optimizer,
            max_iters=cfg.train.max_iter,
            warmup_iters=scheduler_config.get('warmup_iters', 0),
            warmup_factor=scheduler_config.get('warmup_factor', 0.001),
            delay_iters=scheduler_config.get('delay_iters', 0),
            decay_steps=scheduler_config.get('decay_steps', []),
            peak_scale=scheduler_config.get('peak_scale', 0.8),
            eta_min=scheduler_config.get('eta_min', 0.05)
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


def plot_lr_scheduler(optimizer, scheduler, total_iters=100000, save_dir=''):
    """
    模拟训练过程，绘制学习率随迭代步数变化的曲线。

    Args:
        optimizer: 优化器实例（用于获取初始学习率）
        scheduler: 学习率调度器实例
        total_iters: 总迭代步数（通常设为 cfg.train.max_iter）
        save_dir: 图像保存目录（若为空则显示图像）
    """
    opt = deepcopy(optimizer)
    sched = deepcopy(scheduler)
    
    lrs = []
    steps = []
    for it in range(total_iters):
        sched.step()
        current_lr = sched.optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        steps.append(it)
    
    plt.figure(figsize=(12, 5))
    plt.plot(steps, lrs, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, total_iters)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / 'lr_schedule.png', dpi=200, bbox_inches='tight')
        print(f"LR curve saved to {save_dir}/lr_schedule.png")
    else:
        plt.show()
    plt.close()
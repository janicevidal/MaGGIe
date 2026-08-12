import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

def plot_lr_scheduler(optimizer, scheduler, total_iters=100000, save_dir=''):
    """
    模拟训练过程，绘制学习率随迭代步数变化的曲线。

    Args:
        optimizer: 优化器实例（用于获取初始学习率）
        scheduler: 学习率调度器实例
        total_iters: 总迭代步数（通常设为 cfg.train.max_iter）
        save_dir: 图像保存目录（若为空则显示图像）
    """
    # 深拷贝避免修改原始对象
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
    # 如果 save_dir 非空则保存，否则显示
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_dir) / 'lr_schedule.png', dpi=200, bbox_inches='tight')
        print(f"LR curve saved to {save_dir}/lr_schedule.png")
    else:
        plt.show()
    plt.close()
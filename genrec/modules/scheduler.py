"""
Metrics
"""
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class InverseSquareRootScheduler(LRScheduler):
    """
    InverseSquareRootScheduler
    """
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        """
        Initialize the InverseSquareRootScheduler
        """
        self.warmup_steps = warmup_steps
        super(InverseSquareRootScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Get the learning rate.
        Linear warmup from 0 to base_lr, then inverse square root decay.
        """
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        scale_factor = (self.warmup_steps ** 0.5) / (step ** 0.5)
        return [base_lr * scale_factor for base_lr in self.base_lrs]
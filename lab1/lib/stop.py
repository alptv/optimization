from .linesearch import StopCondition, Statistics
import numpy as np
    
class L2StepSizeStop(StopCondition):
    def __init__(self, it_count=None, eps=0.001) -> None:
        super().__init__()
        self.eps = eps
        self.it_count = it_count
        self.it = 0
    def stop(self, f, x, step) -> tuple:
        if self.it_count is not None:
            self.it += 1
        return (self.it_count is not None and self.it > self.it_count) or np.sqrt(np.sum(step ** 2)) < self.eps, Statistics()
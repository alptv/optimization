from abc import ABC, abstractmethod
from collections import Counter
import numpy as np


class DirectionProvider(ABC):
    @abstractmethod
    def direction(self, f, x) -> tuple:
        pass


class Scheduler(ABC):
    @abstractmethod
    def lr(self, f, x, direction) -> tuple:
        pass


class StopCondition(ABC):
    @abstractmethod
    def stop(self, f, x, step) -> tuple:
        pass


class LineSearch:
    def __init__(self, direction_provider: DirectionProvider,
                 scheduler: Scheduler, stop_condition: StopCondition):
        self.direction_provider = direction_provider
        self.scheduler = scheduler
        self.stop_condition = stop_condition

    def search_min(self, x0, f):
        x = x0
        stat = Statistics()
        history = np.array([x])
        while True:
            direction, direction_stat = self.direction_provider.direction(f, x)
            lr, lr_stat = self.scheduler.lr(f, x, direction)
            step = lr * direction
            stop, stop_stat = self.stop_condition.stop(f, x, step)
            if stop:
                break
            x = x + step
            history = np.vstack([history, x])
            stat = stat + direction_stat + lr_stat + stop_stat
            stat.add(Statistics.STEPS)
        return x, stat, history
    
class Statistics:
    GRADIENT = "grad"
    FUNCTION = "f"
    STEPS = "steps"

    def __init__(self, source=None) -> None:
        self.stat = {} if source is None else source

    def get(self, key):
        if key in self.stat:
            return self.stat[key]
        else:
            return 0
        
    def add(self, key, amount=1):
        self.stat[key] = self.get(key) + amount
        
    def toMap(self):
        return self.stat
        
    def __add__(self, other):
        return Statistics(dict(Counter(self.stat) + Counter(other.stat)))
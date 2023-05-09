from .linesearch import Scheduler, Statistics
from .grad import grad
import numpy as np


class ConstantScheduler(Scheduler):
    def __init__(self, step_value):
        super().__init__()
        self.step_value = step_value

    def lr(self, f, x, direction) -> tuple:
        return self.step_value, Statistics()


class GoldenSearchScheduler(Scheduler):
    PART = (3 - np.sqrt(5)) / 2

    def __init__(self, step_value=0.1, eps=0.001):
        super().__init__()
        self.eps = eps
        self.step_value = step_value

    def lr(self, f, x, direction) -> tuple:
        def f_value(alpha):
            return f(x + alpha * direction)

        def left_point(l, r):
            p = l + GoldenSearchScheduler.PART * (r - l)
            return p, f_value(p)

        def right_point(l, r):
            p = r - GoldenSearchScheduler.PART * (r - l)
            return p, f_value(p)

        stat = Statistics()
        l = 0
        r = self.step_value
        a, fa = left_point(l, r)
        b, fb = right_point(l, r)
        stat.add(Statistics.FUNCTION, 2)
        while r - l > self.eps:
            if fa < fb:
                r = b
                b, fb = a, fa
                a, fa = left_point(l, r)
            else:
                l = a
                a, fa = b, fb
                b, fb = right_point(l, r)
            stat.add(Statistics.FUNCTION)
        return (r + l) / 2, stat


class WolfeScheduler(Scheduler):
    INF = "INF"

    def __init__(self, c1, c2, eps=0.001) -> None:
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.eps = eps

    def lr(self, f, x, direction):
        stat = Statistics()
        stat.add(Statistics.FUNCTION)
        stat.add(Statistics.GRADIENT)

        fx = f(x)
        df = grad(f, x) @ direction

        step = 1
        step_min = 0
        step_max = WolfeScheduler.INF

        while True:
            if step_max != WolfeScheduler.INF and step_max - step_min < self.eps:
                return step, stat


            stat.add(Statistics.FUNCTION)
            satisfies_sufficient_decrease = f(x + step * direction) <= fx + self.c1 * step * df
            if not satisfies_sufficient_decrease:
                step_max = step
                step = self.next_step(step, step_min, step_max)
                continue


            stat.add(Statistics.GRADIENT)
            satisfies_curvative_condition = grad(f, x + step * direction) @ direction >= self.c2 * df
            if not satisfies_curvative_condition:
                step_min = step
                step = self.next_step(step, step_min, step_max)
                continue

            return step, stat

    def next_step(self, step, step_min, step_max):
        return step * 2 if (
            step_max == WolfeScheduler.INF) else (step_max + step_min) / 2

from lib.linesearch import LineSearch
from lib.direction import GradientDirection
from lib.stop import L2StepSizeStop
from lib.scheduler import ConstantScheduler, GoldenSearchScheduler, WolfeScheduler


def constant_step_search(step, max_it=10000):
    return LineSearch(GradientDirection(),  ConstantScheduler(step), L2StepSizeStop(it_count=max_it, eps=0.0001))

def golden_search(step, max_it=10000):
    return LineSearch(GradientDirection(), GoldenSearchScheduler(step), L2StepSizeStop(it_count=max_it, eps=0.0001))

def wolfe_search(c1=0.0001, c2=0.1, max_it=10000):
    return LineSearch(GradientDirection(), WolfeScheduler(c1, c2), L2StepSizeStop(it_count=max_it, eps=0.0001))
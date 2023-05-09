import numpy as np

def grad(f, x, eps=0.001):
    fx = f(x)
    n = x.shape[0]
    def delta(i):
        h = np.zeros(n)
        h[i] += eps
        return h
    grad = np.zeros(n)
    for i in range(n): 
        grad[i] = (f(x + delta(i)) - fx) / eps
    return grad
import numpy as np

def generate_quadratic(n, cond_number):
    b = np.random.rand(n)
    diagonal = np.random.uniform(low=1 / cond_number, high=1, size=(n,))
    diagonal[0] = 1 / cond_number
    diagonal[n - 1] = 1
    A = np.diag(diagonal)
    return lambda x: x @ A @ x  - b @ x
import numpy as np


class SumFunctionalModel():
    def __init__(self, n, functions, grads, point) -> None:
        self.n = n
        self.functions = functions
        self.grads = grads
        self.X = np.arange(0, len(functions))
        self.Y = self.X
        self.point = point.copy()
        self.grad_point = None

    def update(self, batch_index, _):
        k = len(batch_index)
        self.grad_point = np.zeros(self.n)
        for i in batch_index:
            self.grad_point += self.grads[i](self.point) / k
        return None

    def grad(self):
        return (self.grad_point,)

    def params(self):
        return (self.point,)


class LinearRegression():
    def __init__(self, n, w=None, b=None) -> None:
        self.w = np.zeros((n, 1)) if w is None else w
        self.b = np.zeros((1, 1)) if b is None else b
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b)
    
    def update(self, X, Y):
        diff = (X @ self.w + self.b - Y)
        mean_diff_doubled = 2 * diff / X.shape[0]
        self.w_grad = X.T @ mean_diff_doubled
        self.b_grad = np.sum(mean_diff_doubled)
        return np.sum(diff ** 2) / X.shape[0]
    
    def predict(self, X):
        return X @ self.w + self.b

    def grad(self):
        return self.w_grad, self.b_grad

    def params(self):
        return self.w, self.b
    


class PolynomialRegreesion1D():
    def __init__(self, degree, reg_coef=0, w=None) -> None:
        self.degree = degree
        self.w = np.zeros((degree + 1, 1)) if w is None else w
        self.w_grad = np.zeros_like(self.w)
        self.reg_coef = reg_coef

        
    def update(self, X, Y):
        polynomial = self.polynomial(X)
        diff = polynomial @ self.w - Y.reshape((-1, 1))
        mean_diff_doubled = 2 * diff / X.shape[0]
        self.w_grad = polynomial.T @ mean_diff_doubled + self.reg_coef * 2 * self.w
        return np.sum((diff / np.sqrt(X.shape[0])) ** 2) + self.reg_coef * np.sum(self.w ** 2)
    
    def predict(self, X):
        return (self.polynomial(X) @ self.w).ravel()

    def grad(self):
        return self.w_grad,

    def params(self):
        return self.w,
    
    def polynomial(self, X):
        return np.vander(X, N=self.degree + 1, increasing=True)
import numpy as np



class ConstantStepOptimizer():
    def __init__(self, alpha, model) -> None:
        self.model = model
        self.alpha = alpha

    def step(self, X, Y):
        self.model.update(X, Y)
        for grad, param in zip(self.model.grad(), self.model.params()):
            param -= self.alpha * grad


class DecayStepOptimizer():
    def __init__(self, alpha, decay, model) -> None:
        self.model = model
        self.alpha = alpha
        self.decay = decay

    def step(self, X, Y):
        self.model.update(X, Y)
        for grad, param in zip(self.model.grad(), self.model.params()):
            param -= self.alpha * grad
        self.alpha = max(0.001, self.alpha - self.decay)


class Momentum():
    def __init__(self, alpha, gamma, model) -> None:
        self.v = [np.zeros_like(param) for param in model.params()]
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def step(self, X, Y):
        self.model.update(X, Y)
        for grad, param, v in zip(self.model.grad(), self.model.params(), self.v):
            v *= self.gamma
            v += (1 - self.gamma) * grad
            param -= self.alpha * v

class Nesterov():
    def __init__(self, alpha, gamma, model) -> None:
        self.v = [np.zeros_like(param) for param in model.params()]
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def step(self, X, Y):
        for param, v in zip(self.model.params(), self.v):
            param -= self.alpha * self.gamma * v
        self.model.update(X, Y)
        for grad, param, v in zip(self.model.grad(), self.model.params(), self.v):
            param += self.alpha * self.gamma * v
            v *= self.gamma
            v += (1 - self.gamma) * grad 
            param -= self.alpha * v

class AdaGrad():
    def __init__(self, alpha, model) -> None:
        self.G = [np.zeros_like(param) for param in model.params()]
        self.alpha = alpha
        self.model = model
        
    def step(self, X, Y):
        self.model.update(X, Y)
        for grad, param, G in zip(self.model.grad(), self.model.params(), self.G):
            G += grad * grad
            param -= self.alpha * grad / np.sqrt(G)


class RMSProp():
    def __init__(self, alpha, gamma, model) -> None:
        self.s = [np.zeros_like(param) for param in model.params()]
        self.alpha = alpha
        self.model = model
        self.gamma = gamma
        self.eps = 1e-2
        
    def step(self, X, Y):
        self.model.update(X, Y)
        for grad, param, s in zip(self.model.grad(), self.model.params(), self.s):
            s *= self.gamma
            s +=  (1 - self.gamma) * grad * grad
            param -= self.alpha * grad / np.sqrt(s + self.eps)

class Adam():
    def __init__(self, alpha, b1, b2, model) -> None:
        self.v = [np.zeros_like(param) for param in model.params()]
        self.s = [np.zeros_like(param) for param in model.params()]
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.model = model
        self.eps = 1e-4
        self.it = 0
        
    def step(self, X, Y):
        self.it += 1
        self.model.update(X, Y)
        for grad, param, v, s in zip(self.model.grad(), self.model.params(), self.v, self.s):
                v *= self.b1
                v += (1 - self.b1) * grad
                s *= self.b2
                s += (1 - self.b2) * grad * grad
                v_c = v / (1 - self.b1 ** self.it)
                s_c = s / (1 - self.b2 ** self.it)
                param -= self.alpha * v_c / np.sqrt(s_c + self.eps)
        
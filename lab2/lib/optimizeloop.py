import numpy as np
from lib.model import SumFunctionalModel, LinearRegression
    

def optimize_sum_functions(n, functions, grads, point, optimizer_provider, batch_size, max_epoch=2000,eps=0.001, trace_history=False):
    model = SumFunctionalModel(n, functions, grads, point)
    X = model.X
    Y = model.Y
    return (model.point, ) + optimize_model(model, optimizer_provider(model), X, Y, batch_size, max_epoch, eps, trace_history=trace_history)
    

def optimize_model(model, optimizer, X, Y, batch_size, max_epoch=200,eps=0.001, trace_history=False):
    X, Y = shuffle(X, Y)
    data_size = X.shape[0]
    history = [model.point.copy()] if trace_history else None
    params = copy_params(model)
    it = 0
    for epoch in range(1, max_epoch + 1):
        for batch_start in range(0, data_size, batch_size):
            batch_end = batch_start + batch_size
            optimizer.step(X[batch_start:batch_end], Y[batch_start:batch_end])
            if trace_history:
                history.append(model.point.copy())
            if params_max_diff(model.params(), params) < eps:
                return result(epoch, it, history)
            params = copy_params(model)
            it += 1

    return result(epoch, it, history)

def result(epoch, it, history):
    if history is None:
        return epoch, it
    else: 
        return epoch, it, history

def copy_params(model):
    model_params = model.params()
    copied_params = [None for _ in model_params]
    for i in range(len(copied_params)):
        copied_params[i] = model_params[i].copy()
    return copied_params

def params_max_diff(prev_params, params):
    return max([np.abs(param - prev_param).max() for param, prev_param, in zip(params, prev_params)])
        
def shuffle(X, Y):
    p = np.random.permutation(len(Y))
    return X[p], Y[p]
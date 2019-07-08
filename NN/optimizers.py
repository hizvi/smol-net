import numpy as np

class Optimizer:
    def __init__(self):
        pass
    
    def get_updates(self, network, gradients):
        raise NotImplementedError('This is an abstract Optimizer class')


class SGD(Optimizer):
    def __init__(self, lr=0.1, decay=1e-6):
        self.lr = lr
        self.decay = decay
        self.iterations = 0
        
    def get_updates(self, network, gradients):
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        for layer, grad in zip(network.layers, gradients):
            layer.weights -= lr * grad
        self.iterations += 1
        
        
def get_optimizer(name='sgd'):
    
    optimizers = {
        'sgd':  SGD,
        # add elements to dict as optimizers grow
    }
    
    if isinstance(name, str):
        if name not in optimizers:
            raise ValueError('Invalid optimizer specified')

        return optimizers[name]
    else: 
        return name




##  POSSIBLE FUTURE WORK

class SomeOtherOptimzer():
    def __init__(self):
        raise NotImplementedError

    def get_updates(self):
        raise NotImplementedError
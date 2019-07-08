import numpy as np

from . import Activations

class Layer:
    # TODO: define what a layer looks like
    def __init__(self, units):
        self.units = units
        self.indim = None
        self.outdim = units

    
    def weight_initializer(self, shape):
        raise NotImplementedError("Choose appropriate layer: Layer is an abstract class.")
        

class Input(Layer):
    def __init__(self, input_dims):
        super(Input, self).__init__(input_dims)
    
    
class FullyConnected(Layer):
    # TODO: implement a fully connected feed forward layer.
    def __init__(self, units, activation='sigmoid', predefined_weights=None):
        super(FullyConnected, self).__init__(units)
        
        if predefined_weights:
            assert predefined_weights.shape[0] == units, 'predefined weights have invalid dimensions, predefined_weights.shape[0] != number of units'
            self.weights = predefined_weights
        
        self.activation, self.activation_dx = Activations.get_activation(activation)

    
    def forward(self, X):
        # forward pass
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.WX = self.X.dot(self.weights)

        return self.activation(self.WX)
        
    def backward(self, incoming):
        # backward pass
        if self.activation_dx:
            incoming *= self.activation_dx(self.WX)
        dW = np.dot(self.X.T, incoming)
        dX = np.dot(incoming, self.weights[1:].T)

        return dX, dW        

    def weight_initializer(self, *shape):
        # initialize weights ... during compile time
        self.weights = np.random.rand(*shape) / 100

##  POSSIBLE FUTURE WORK

# TODO: other layers
class SomeOtherLayer(Layer):
    def __init__(self):
        raise NotImplementedError()
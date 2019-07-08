import numpy as np

class Activations:
    # linear activation (no activation)
    @staticmethod
    def linear(x):
        return x

    # sigmoid derivative
    @staticmethod
    def linear_derivative(x):
        return np.ones(x.shape)

    
    # sigmoid activation function
    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-1 * x))

    # sigmoid derivative
    @staticmethod
    def sigmoid_derivative(x):
        sigmoid = Activations.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    # softmax activation function
    @staticmethod
    def softmax(x):
        # subtract by x.max to avoid overflow
        e_x = np.exp(x - x.max(axis=1).reshape(-1,1))
        return e_x / e_x.sum(axis=1).reshape(-1,1)

    ##  POSSIBLE FUTURE WORK

    # TODO: relu activation function
    @staticmethod
    def relu(x):
        raise NotImplementedError

    @staticmethod
    def relu_derivative(x):
        raise NotImplementedError

    # TODO: tanh activation function
    @staticmethod
    def tanh(x):
        raise NotImplementedError

    @staticmethod
    # getter method: maps activation function identifier to function 
    def get_activation(name='sigmoid'):
        
        activations = {
            'linear' : (Activations.linear, Activations.linear_derivative),
            'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative), 
            'softmax': (Activations.softmax, None)
            # add elements to dict as activations grow
        }
        
        if name not in activations:
            raise ValueError('Invalid activation function specified')
        
        return activations[name]
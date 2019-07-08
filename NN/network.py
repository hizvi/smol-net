import numpy as np

from . import Losses
from . import Metrics
from . import optimizers

from copy import copy, deepcopy

class NN:
    def __init__(self, input_dim, layers=None):
        self.layers = layers or []
        # self.units = [l.units for l in layers] or []
        self.layer_count = len(self.layers)
        self.input_dim = input_dim


    def add(self, layer):
        self.layers.append(layer)

    def _forward(self, X):
        Z = deepcopy(X)
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def compile(self, optimizer='sgd', loss='categorical_crossentropy', metric='accuracy'):
        # set up loss, metrics and optimizer
        self.loss = Losses.get_loss(loss)
        self.metric = Metrics.get_metric(metric)
        self.optimizer = optimizers.get_optimizer(optimizer)()
        
        # initialize weights for each layer
        input_dim = self.input_dim
        for layer in self.layers:
            layer.weight_initializer(input_dim + 1, layer.units)
            input_dim = layer.units
            

    def predict(self, X):
        assert hasattr(self, 'loss') and hasattr(self, 'optimizer'), 'configure model first using compile() before calling fit'
        return np.argmax(self._forward(X), axis=-1).reshape(-1, 1)
    
    def evaluate(self, X, y):
        # predict and evaluate
        return self.metric(y, self.predict(X))
    
    
    def fit(self, X, y, epochs=1, batch_size=32):
        # make sure model is compiled before fitting
        assert hasattr(self, 'loss') and hasattr(self, 'optimizer'), 'configure model first using compile() before calling fit'
        assert X.shape[0] == y.shape[0], 'Inconsistent dimensions of training data, X.shape[0] != y.shape[0]'
        
        training_metrics = []
        training_loss = []
        
        for epoch in range(epochs):
            # shuffle datapoints
            shuffled_idx = np.random.permutation(X.shape[0])
            X, y = X[shuffled_idx], y[shuffled_idx]
            # steps_per_epoch = max(1, X.shape[0] // batch_size)
            
            # make mini batches 
            batches = [
                (X[step: step + batch_size], y[step: step + batch_size]) 
                       for step in range(0, X.shape[0], batch_size)
            ]
            
            # fit the mini batch
            for X_batch, y_batch in batches:
                self.fit_batch(X_batch, y_batch)
            
            # record losses and metrics for each epoch
            y_pred = self._forward(X)
            loss = self.loss(y, y_pred)
            metric = self.metric(y, y_pred)
            
            # print(y_pred)
            
            training_loss.append(loss)
            training_metrics.append(metric)

            print('EPOCH {}/{}, loss: {}, accuracy: {}'.format(epoch + 1, epochs, loss, metric))

        return {
            'loss': training_loss,
            'metric': training_metrics,
        }


    def fit_batch(self, X_batch, y_batch):
        grads = []
        y_batch_pred = self._forward(X_batch)

        dX = y_batch_pred - y_batch

        # backpropagation
        for layer in self.layers[::-1]:
            dX, dW = layer.backward(dX)
            grads.append(dW)
        
        # update weights
        self.update_gradients(reversed(grads))


    def update_gradients(self, gradients):
        # update the weights for each layer in the network.
        self.optimizer.get_updates(self, gradients)  
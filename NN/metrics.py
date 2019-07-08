import numpy as np

class Metrics:
    @staticmethod
    def accuracy(y_true , y_pred):
        assert y_true.shape == y_pred.shape
        
        if y_pred.shape[-1] == 1:
            return Metrics.binary_accuracy(y_true , y_pred)
        else:
            return Metrics.categorical_accuracy(y_true , y_pred)
        
    @staticmethod
    def binary_accuracy(y_true , y_pred):
        return np.mean(np.equal(y_true, y_pred)) # axis=-1
        
    @staticmethod    
    def categorical_accuracy(y_true, y_pred):
        return np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1))) # axis=-1

    @staticmethod
    def get_metric(name='accuracy'):
        
        metrics = {
            'accuracy':  Metrics.accuracy,
            # add elements to dict as metrics grow
        }
        
        if name not in metrics:
            raise ValueError('Invalid metric specified')
        
        return metrics[name]



    ##  POSSIBLE FUTURE WORK
    # TODO: other 
    @staticmethod
    def some_other_metric(y_true , y_pred):
        raise NotImplementedError



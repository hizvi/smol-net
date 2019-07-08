import numpy as np

class Losses:
    @staticmethod
    def binary_crossentropy(y_true , y_pred):
        raise NotImplementedError()
        
    @staticmethod    
    def categorical_crossentropy(y_true, y_pred):
        return -1 * np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    ##  POSSIBLE FUTURE WORK
    # TODO: other 
    @staticmethod
    def some_other_loss(y_true , y_pred):
        raise NotImplementedError()

    @staticmethod
    def get_loss(name='categorical_crossentropy'):
        
        losses = {
            'categorical_crossentropy':  Losses.categorical_crossentropy,
            'mse': Losses.mse
            # add elements to dict as losses grow
        }
        
        if name not in losses:
            raise ValueError('Invalid loss specified')
        
        return losses[name]


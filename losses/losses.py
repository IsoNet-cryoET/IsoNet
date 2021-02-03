
import tensorflow.keras.backend as K

def mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)

def loss_mae(mean=True):
    R = mean_or_not(mean)
    def mae(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.abs(y_pred[...,:n] - y_true))
    return mae

def loss_mse(mean=True):
    R = mean_or_not(mean)
    def mse(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.square(y_pred[...,:n] - y_true))
    return mse

def new_mae(y_true, y_pred):
    n = K.shape(y_true)[-1]
    np_true = y_true.numpy()
    return K.abs(y_pred[...,:n] - y_true)

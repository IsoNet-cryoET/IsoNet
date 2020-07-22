from utils import mean_or_not

def loss_mse(mean=True):
    R = mean_or_not(mean)
    def mse(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.square(y_pred[...,:n] - y_true))
    return mse
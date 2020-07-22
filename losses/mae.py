from utils import mean_or_not

def loss_mae(mean=True):
    R = mean_or_not(mean)
    def mae(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.abs(y_pred[...,:n] - y_true))
    return mae
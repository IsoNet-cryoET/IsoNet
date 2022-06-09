import tensorflow as tf
import numpy as np
from IsoNet.preprocessing.simulate import TwoDPsf
import mrcfile

# def loss_wedge_mae(y_true,y_pred):
    
def wedge_power_gain(y_true,y_pred):

    alpha = 0.15
    y_pred_np = y_pred.numpy() # in the shape of [chanels,side,side,side,n]
    N = y_pred_np.shape[-1]
    cube_list = [np.squeeze(y_pred_np,axis=0)[:,:,:,i] for i in range(N)]
    loss = list(map(wedge_power_ratio,cube_list))
    loss_arr = np.array(loss)
    wedge_loss = 1-tf.convert_to_tensor(loss_arr)
    mae = tf.keras.backend.mean(tf.math.abs(y_pred - y_true), axis=-1)
    total_loss = mae + alpha*mae
    return total_loss



def wedge_power_ratio(tomo):
    sidelen = tomo.shape[0]
    mw = TwoDPsf(sidelen,sidelen).getMW()
    cirle = TwoDPsf(sidelen,sidelen).circleMask()
    ny = tomo.shape[1]
    y_ave = np.zeros([sidelen,sidelen])
    for i in range(ny):
        y_sli = tomo[:,i,:]
        y_sli_norm = (y_sli - np.mean(y_sli))/np.std(y_sli)
        y_sli_fft = np.fft.fftshift(np.fft.fft2(y_sli_norm))
        y_ave = y_ave + np.abs(y_sli_fft)
    y_ave_wedge = np.clip((cirle-mw),0,1)*y_ave
    y_ave_base = mw*y_ave
    return power(y_ave_wedge)/power(y_ave_base)

def power(xz):
    num = np.where(xz > 1)[0].shape[0] +1
    print(num)
    mean_power = np.sum(xz)/num
    return mean_power

if __name__ == '__main__':
    import time 
    sidelen = 96
    mw = np.repeat(TwoDPsf(sidelen,sidelen).getMW()[:,np.newaxis,:],sidelen,axis=1)
    circle = np.repeat(TwoDPsf(sidelen,sidelen).circleMask()[:,np.newaxis,:],sidelen,axis=1)
    gain = (circle -mw)/2
    with mrcfile.open('/storage/heng/mwrtest3D/t371_2/cuberesults/pp676-bin4-wbp_000073_iter15.mrc') as op:
        data = op.data
    a = tf.convert_to_tensor(mw[np.newaxis,:,:,:,np.newaxis])
    b = tf.convert_to_tensor(data[np.newaxis,:,:,:,np.newaxis])
    st = time.time()
    print(wedge_power_gain(a,b))
    ed = time.time()
    print('run time: ',ed-st)
    
    

    

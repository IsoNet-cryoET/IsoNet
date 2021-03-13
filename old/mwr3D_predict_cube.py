#!/usr/bin/env python3
import mrcfile
import tensorflow as tf
from tensorflow.keras.models import Model
from mwr.util.norm import normalize

def predict(settings):
    from tensorflow.keras.models import model_from_json
    json_file = open(settings.model, 'r')
    loaded_model_json = json_file.read()

    json_file.close()
    model = model_from_json(loaded_model_json)

    if settings.ngpus >1:
        from tensorflow.keras.utils import multi_gpu_model
        model = Model(model, gpus=settings.ngpus, cpu_merge=True, cpu_relocation=False)
    model.load_weights(settings.weight)
    with mrcfile.open(settings.mrcfile) as f:
        dat = f.data
    real_data=normalize(dat, percentile = settings.normalize_percentile)

    cube_size = real_data.shape[0]
    pad_size1 = (settings.predict_cropsize - cube_size)//2
    pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
    padi = (pad_size1,pad_size2)
    real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')
    predicted=model.predict(real_data[np.newaxis,:,:,:,np.newaxis], batch_size=1, verbose=1)
    end_size = pad_size1+cube_size
    outData = predicted.reshape(predicted.shape[1:-1])
    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
    with mrcfile.new(settings.out_name) as n:
        n.set_data(outData1)
        
        
class Setting():
    def __init__(self):
        self.mrcfile = '/storage/heng/polyribo/train/t231/noise64/n_00001.mrc'
        self.out_name = '/storage/heng/polyribo/train/t231/n_00001_iter20.mrc'
        self.weight =   '/storage/heng/polyribo/train/t231/results/model_iter20.h5'
        self.model = '/storage/heng/polyribo/train/t231/model.json'
        self.predict_cropsize = 96
        self.normalize_percentile = True
        self.ngpus = 1


if __name__ == '__main__':
    settings = Setting()    
    predict(settings)
    

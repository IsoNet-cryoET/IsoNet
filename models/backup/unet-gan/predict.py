import torch
#import logging
#tf.get_logger().setLevel(logging.ERROR)
#from tensorflow.keras.models import load_model
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
#import tensorflow.keras.backend as K
#import tensorflow as tf

def predict(settings):    
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    #strategy = tf.distribute.MirroredStrategy()
    #if settings.ngpus >1:
    #    with strategy.scope():
    #        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    #else:
    #    model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    from .model import Unet,Context_encoder
    from torch.utils.data import DataLoader
    model = Context_encoder().cuda()
    from .train import reload_ckpt
    reload_ckpt('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1), model)
    model = torch.nn.DataParallel(model)
    model.eval()
    from .data_sequence import Predict_sets
    bench_dataset = Predict_sets(settings.mrc_list)
    bench_loader = DataLoader(bench_dataset, batch_size=4, num_workers=1)
    predicted = []
    with torch.no_grad():
        for idx, val_data in enumerate(bench_loader):
            res=model(val_data['image']).cpu().detach().numpy().astype(np.float32)
            print(res.shape)
            for item in res:
                it = item.squeeze(0)
                predicted.append(it)
    
    for i,mrc in enumerate(settings.mrc_list):
        root_name = mrc.split('/')[-1].split('.')[0]
        outData = normalize(predicted[i], percentile = settings.normalize_percentile)
        with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
            output_mrc.set_data(-outData) 
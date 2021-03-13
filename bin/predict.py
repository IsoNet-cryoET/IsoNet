#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging

from IsoNet.preprocessing.simulate import apply_wedge
from IsoNet.util.norm import normalize
from IsoNet.util.toTile import reform3D
import mrcfile
from IsoNet.util.image import *
from tensorflow.keras.models import load_model
    

def predict(args):
    # if args.log_level == 'debug':
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # else:
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S')
    if args.log_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID 
    logger.debug('percentile:{}'.format(args.norm))

    ngpus = len(args.gpuID.split(','))
    logger.info('gpuID:{}'.format(args.gpuID))
    if ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = load_model(args.model)
    else:
        model = load_model(args.model)

    logger.info("Loaded model from disk")

    if os.path.isfile(args.mrc_file):
        predict_one(args,args.mrc_file,model,output_file=args.output_file)
    if os.path.isdir(args.mrc_file):
        for tomo_str in os.listdir(args.mrc_file):
            predict_one(args,args.mrc_file+'/'+tomo_str,model)


def predict_one(args,one_tomo,model,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    ngpus = len(args.gpuID.split(','))
    N = args.batch_size * ngpus
    root_name = one_tomo.split('/')[-1].split('.')[0]
    if output_file is None:
        if os.path.isdir(args.output_file):
            output_file = args.output_file+'/'+root_name+'_corrected.mrc'
        else:
            output_file = root_name+'_corrected.mrc'
    else:
        pass
    print('predicting:{}'.format(root_name))
    with mrcfile.open(one_tomo) as mrcData:
        real_data = mrcData.data.astype(np.float32)*-1
    real_data = normalize(real_data,percentile=args.norm)
    data=np.expand_dims(real_data,axis=-1)
    reform_ins = reform3D(data)
    data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
    #imposing wedge to every cubes
    #data=wedge_imposing(data)

    num_batches = data.shape[0]
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = np.append(data, data[0:append_number], axis = 0)

    outData=model.predict(data, batch_size= args.batch_size,verbose=1)

    outData = outData[0:num_batches]
    outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

    outData = normalize(outData,percentile=args.norm)
    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(-outData)
    print('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)
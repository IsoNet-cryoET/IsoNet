import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.models import load_model
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
import tensorflow.keras.backend as K
import os
from IsoNet.util.toTile import reform3D
from tqdm import tqdm
import sys

def predict(settings):    
    # model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))
    strategy = tf.distribute.MirroredStrategy()
    if settings.ngpus >1:
        with strategy.scope():
            model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    else:
        model = load_model('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count-1))
    N = settings.predict_batch_size 
    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []
    for i,mrc in enumerate(list(settings.mrc_list) + list(settings.mrc_list[:append_number])):
        root_name = mrc.split('/')[-1].split('.')[0]
        with mrcfile.open(mrc) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)
            predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=0)
            predicted = predicted.reshape(predicted.shape[0:-1])
            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count-1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []
    K.clear_session()
    
def predict_one(args,one_tomo,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    import logging
    if args.ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(args.model)
    else:
        model = tf.keras.models.load_model(args.model)

    logging.info("Loaded model from disk")
    root_name = one_tomo.split('/')[-1].split('.')[0]

    if output_file is None:
        if os.path.isdir(args.output_file):
            output_file = args.output_file+'/'+root_name+'_corrected.mrc'
        else:
            output_file = root_name+'_corrected.mrc'

    logging.info('predicting:{}'.format(root_name))

    with mrcfile.open(one_tomo) as mrcData:
        real_data = mrcData.data.astype(np.float32)*-1
        voxelsize = mrcData.voxel_size
    real_data = normalize(real_data,percentile=args.normalize_percentile)
    data=np.expand_dims(real_data,axis=-1)
    reform_ins = reform3D(data)
    data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
    #imposing wedge to every cubes
    #data=wedge_imposing(data)

    N = args.batch_size #* args.ngpus * 4 # 8*4*8
    num_patches = data.shape[0]
    if num_patches%N == 0:
        append_number = 0
    else:
        append_number = N - num_patches%N
    data = np.append(data, data[0:append_number], axis = 0)
    num_big_batch = data.shape[0]//N
    outData = np.zeros(data.shape)
    logging.info("total batches: {}".format(num_big_batch))
    for i in tqdm(range(num_big_batch), file=sys.stdout):
        in_data = data[i*N:(i+1)*N]
        # in_data_gen = get_gen_single(in_data,args.batch_size,shuffle=False)
        # in_data_tf = tf.data.Dataset.from_generator(in_data_gen,output_types=tf.float32)
        outData[i*N:(i+1)*N] = model.predict(in_data,verbose=0)
    outData = outData[0:num_patches]

    outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

    outData = normalize(outData,percentile=args.normalize_percentile)
    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(-outData)
        output_mrc.voxel_size = voxelsize
    K.clear_session()
    logging.info('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)

#!/usr/bin/env python3
import numpy as np
import os, sys
from IsoNet.preprocessing.simulate import apply_wedge
from IsoNet.util.norm import normalize
from IsoNet.util.toTile import reform3D
import mrcfile
from IsoNet.util.image import *
from IsoNet.util.metadata import MetaData,Label,Item
from IsoNet.util.dict2attr import idx2list
from tqdm import tqdm
# from IsoNet.training.data_sequence import get_gen_single
def predict(args):

    if args.log_level == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    import logging
    tf_logger = tf.get_logger()
    tf_logger.setLevel(logging.ERROR)

    logger = logging.getLogger('predict')
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    logging.info('\n\n######Isonet starts predicting######\n')

    args.gpuID = str(args.gpuID)
    args.ngpus = len(list(set(args.gpuID.split(','))))
    if args.batch_size is None:
        args.batch_size = max(4, 2 * args.ngpus)
    #print('batch_size',args.batch_size)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    #check gpu settings
    from IsoNet.bin.refine import check_gpu
    check_gpu(args)

    logger.debug('percentile:{}'.format(args.normalize_percentile))

    logger.info('gpuID:{}'.format(args.gpuID))
    if args.ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(args.model)
    else:
        model = tf.keras.models.load_model(args.model)

    logger.info("Loaded model from disk")

    if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        # write star percentile threshold
    md = MetaData()
    md.read(args.star_file)
    if not 'rlnCorrectedTomoName' in md.getLabels():
        md.addLabels('rlnCorrectedTomoName')
        for it in md:
            md._setItemValue(it,Label('rlnCorrectedTomoName'),None)
    args.tomo_idx = idx2list(args.tomo_idx)
    for it in md:
        if args.tomo_idx is None or str(it.rlnIndex) in args.tomo_idx:
            if args.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and it.rlnDeconvTomoName not in [None,'None']:
                tomo_file = it.rlnDeconvTomoName
            else:
                tomo_file = it.rlnMicrographName
            tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]
            if os.path.isfile(tomo_file):
                tomo_out_name = '{}/{}_corrected.mrc'.format(args.output_dir,tomo_root_name)
                predict_one(args,tomo_file,model,output_file=tomo_out_name)
                md._setItemValue(it,Label('rlnCorrectedTomoName'),tomo_out_name)
        md.write(args.star_file)

def predict_one(args,one_tomo,model,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc
    import logging
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

    N = args.batch_size * args.ngpus * 4 # 8*4*8
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
    logging.info('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)

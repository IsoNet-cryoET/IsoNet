import logging
from mwr.preprocessing.cubes import prepare_cubes
from mwr.preprocessing.img_processing import normalize
from mwr.preprocessing.prepare import prepare_first_iter,get_cubes_list
import glob
import mrcfile
import numpy as np
import glob
import os
import shutil
from mwr.training.train import train_data, prepare_first_model
from mwr.training.predict import predict

def run(args):
    #*******set fixed parameters*******
    args.reload_weight = True
    args.result_dir = 'results'
    args.continue_from = "training"
    args.predict_cropsize = args.crop_size
    args.predict_batch_size = args.batch_size
    args.lr = 0.0004
    args.subtomo_dir = args.result_dir + '/subtomo'
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.INFO)
    logger = logging.getLogger('mwr.preprocessing.prepare')
    # Specify GPU(s) to be used
    args.gpuID = str(args.gpuID)
    args.ngpus = len(args.gpuID.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    #TODO to fix the tensorflow log level
    # if args.log_level == 'debug':
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # else:
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #*************************************
    #****prepare for first iteration******
    if args.continue_iter == 0 or args.pretrained_model is None:
        args = prepare_first_iter(args)
        logging.warning("Done preperation for the first iteration!")
        args.continue_iter = 1
        if args.pretrained_model is not None:
            args.init_model = args.pretrained_model
        else:
            args = prepare_first_model(args)
    else: #mush has pretrained model and continue_iter >0
        args.init_model = args.pretrained_model

    args.mrc_list = os.listdir(args.subtomo_dir)
    args.mrc_list = ['{}/{}'.format(args.subtomo_dir,i) for i in args.mrc_list]

    continue_from_training = not os.path.isfile('{}/model_iter{:0>2d}.h5'.format
    (args.result_dir,args.continue_iter))

    #************************************
    for num_iter in range(args.continue_iter,args.iterations):
        args.iter_count = num_iter
        logger.warning("Start Iteration{}!".format(num_iter))
        if num_iter > args.continue_iter: # set the previous iteration's result model as the init_model 
            args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)
        args.noise_factor = ((num_iter - args.noise_start_iter)//args.noise_pause)+1 if num_iter >= args.noise_start_iter else 0
        logging.info("noise_factor:{}".format(args.noise_factor))

        if continue_from_training:
            try:
                shutil.rmtree(args.data_dir)
            except OSError:
                logging.debug("No previous data folder!")
            logging.info('Maybe stack at getting cube?')
            get_cubes_list(args)
            logging.info("Done getting cubes!")
            logging.info("Start training!")
            history = train_data(args) #train based on init model and save new one as model_iter{num_iter}.h5
            # losses.append(history.history['loss'][-1])
            logging.info("Done training!")
            logging.info("Start cube predicting!")
            predict(args)
            logging.info("Done cube predicting!")
        
        else:
            logging.info("Model for iteration {} exists".format(args.continue_iter))
            logging.info("Start cube predicting!")
            predict(args)
            logging.info("Done cube predicting!")
            continue_from_training = True

        logging.info("Done Iteration{}!".format(num_iter+1))

if __name__ == "__main__":
    from mwr.util.dict2attr import Arg
    arg = {'input_dir': 'subtomo/', 'gpuID': '4,5,6,7', 
    'mask_dir': None, 'noise_dir': None, 'iterations': 50, 
    'data_dir': 'data', 'pretrained_model': './results/model_iter35.h5', 
    'log_level': 'debug', 'continue_training': False, 
    'continue_iter': 36, 'noise_mode': 1, 'noise_level': 0.05, 
    'noise_start_iter': 15, 'noise_pause': 5, 'cube_size': 64, 
    'crop_size': 96, 'ncube': 1, 'preprocessing_ncpus': 16, 
    'epochs': 10, 'batch_size': 8, 'steps_per_epoch': 150, 
    'drop_out': 0.3, 'convs_per_depth': 3, 'kernel': (3, 3, 3),
     'unet_depth': 3, 'filter_base': 32, 'batch_normalization': False, 
     'normalize_percentile': True}
    d_args = Arg(arg)
    print(mrcfile.__file__)
    run(d_args)

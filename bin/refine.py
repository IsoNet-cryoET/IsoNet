import logging
from IsoNet.preprocessing.cubes import prepare_cubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.preprocessing.prepare import prepare_first_iter,get_cubes_list
from IsoNet.util.dict2attr import save_args_json,load_args_from_json
import glob
import mrcfile
import numpy as np
import glob
import os
import sys
import shutil
from IsoNet.util.metadata import MetaData, Item, Label
def run(args):
    if args.continue_from is None:
        run_whole(args)

    else:
        run_continue(args)

def run_whole(args):
    md = MetaData()
    md.read(args.subtomo_star)
    #*******set fixed parameters*******
    args.crop_size = md._data[0].rlnCropSize
    args.cube_size = md._data[0].rlnCubeSize
    args.predict_cropsize = args.crop_size
    args.noise_dir = None
    args.lr = 0.0004

    #*******calculate parameters********
    if args.gpuID is None:
        args.gpuID = '0,1,2,3'
    else:
        args.gpuID = str(args.gpuID)
    if args.iterations is None:
        args.iterations = 30
    args.ngpus = len(args.gpuID.split(','))
    if args.result_dir is None:
        args.result_dir = 'results'
    if args.batch_size is None:
        args.batch_size = max(4, 2 * args.ngpus)
    args.predict_batch_size = args.batch_size
    if args.filter_base is None:
        if md._data[0].rlnPixelSize >15:
            args.filter_base = 32
        else:
            args.filter_base = 64
    if args.steps_per_epoch is None:
        args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
    if len(md) <=0:
        logging.error("Subtomo list is empty!")
        sys.exit(0)
    args.mrc_list = []
    for i,it in enumerate(md):
        if "rlnImageName" in md.getLabels():
            args.mrc_list.append(it.rlnImageName)
    
    # Specify GPU(s) to be used
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logger = logging.getLogger('IsoNet.refine')
    if args.log_level == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #import tensorflow related modules after setting environment
    from IsoNet.training.predict import predict
    from IsoNet.training.train import prepare_first_model, train_data
    args = prepare_first_iter(args)
    logger.info("Done preperation for the first iteration!")
    for num_iter in range(1,args.iterations + 1):        
        args.iter_count = num_iter
        logger.info("Start Iteration{}!".format(num_iter))
        # pretrained_model case
        if args.pretrained_model is not None and num_iter==1:
            shutil.copyfile(args.pretrained_model,'{}/model_iter{:0>2d}.h5'.format(args.result_dir,1))
            # os.system('cp {} {}'.format(args.pretrained_model,args.result_dir+'{}/model_iter{:0>2d}.h5'.format(args.result_dir,continue_iter)))
            logger.info('Use Pretrained model as the output model of iteration 1 and predict subtomograms')
            predict(args)
            continue

        if num_iter ==1:
            args = prepare_first_model(args)
        else:
            args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)
        
        args.noise_factor = ((num_iter - args.noise_start_iter)//args.noise_pause)+1 if num_iter >= args.noise_start_iter else 0
        logging.info("noise_level:{}".format(args.noise_factor*args.noise_level))

        try:
            shutil.rmtree(args.data_folder)     
        except OSError:
            pass
    
        get_cubes_list(args)
        logger.info("Done preparing subtomograms!")
        logger.info("Start training!")
        history = train_data(args) #train based on init model and save new one as model_iter{num_iter}.h5
        # losses.append(history.history['loss'][-1])
        save_args_json(args,args.result_dir+'/refine_iter{:0>2d}.json'.format(num_iter))
        logger.info("Done training!")
        logger.info("Start predicting subtomograms!")
        predict(args)
        logger.info("Done predicting subtomograms!")
        logger.info("Done Iteration{}!".format(num_iter))

def run_continue(continue_args):
    
    #params need to to be recalculated when in continue_from mode
    # [gpuID, batch_size, steps_per_epoch, iterations] 
    args = load_args_from_json(continue_args.continue_from)
    md = MetaData()
    md.read(args.subtomo_star)
    if continue_args.iterations is not None:
        args.iterations = continue_args.iterations
    if continue_args.gpuID is not None:
        args.gpuID = str(continue_args.gpuID)
    args.ngpus = len(args.gpuID.split(','))
    if continue_args.batch_size is not None:
        args.batch_size = continue_args.batch_size
    elif continue_args.gpuID is not None:
        args.batch_size = max(4, 2 * args.ngpus)
    args.predict_batch_size = args.batch_size
    if continue_args.steps_per_epoch is not None:
        args.steps_per_epoch = continue_args.steps_per_epoch
    elif continue_args.gpuID is not None:
        args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
    if len(md) <=0:
        logging.error("Subtomo list is empty!")
        sys.exit(0)
    args.mrc_list = []
    for i,it in enumerate(md):
        if "rlnImageName" in md.getLabels():
            args.mrc_list.append(it.rlnImageName)

    #setting up gpu and logger
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuID)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    logger = logging.getLogger('IsoNet.refine')
    if args.log_level == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    from IsoNet.training.predict import predict
    from IsoNet.training.train import prepare_first_model, train_data
    # start REFINE LOOP
    current_iter = args.iter_count
    for num_iter in range(current_iter,args.iterations + 1):
        args.iter_count = num_iter
        logger.info("Start Iteration{}!".format(num_iter))
        # Predict subtomos at first
        if continue_args.continue_from is not None:
            logger.info('Continue from previous model: {}/model_iter{:0>2d}.h5 of iteration {} and predict subtomograms'.format(args.result_dir,num_iter,num_iter))
            predict(args)
            continue_args.continue_from = None
            continue
        ##
        args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)
        args.noise_factor = ((num_iter - args.noise_start_iter)//args.noise_pause)+1 if num_iter >= args.noise_start_iter else 0
        logging.info("noise_level:{}".format(args.noise_factor*args.noise_level))

        try:
            shutil.rmtree(args.data_folder)     
        except OSError:
            pass
    
        get_cubes_list(args)
        logger.info("Done preparing subtomograms!")
        logger.info("Start training!")
        history = train_data(args) #train based on init model and save new one as model_iter{num_iter}.h5
        # losses.append(history.history['loss'][-1])
        save_args_json(args,args.result_dir+'/refine_iter{:0>2d}.json'.format(num_iter))
        logger.info("Done training!")
        logger.info("Start predicting subtomograms!")
        predict(args)
        logger.info("Done predicting subtomograms!")
        logger.info("Done Iteration{}!".format(num_iter))

    
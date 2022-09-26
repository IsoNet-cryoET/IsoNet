import logging
from IsoNet.preprocessing.prepare import get_cubes_list,get_noise_level, prepare_first_iter
from IsoNet.util.dict2attr import save_args_json,load_args_from_json
import numpy as np
import os
import sys
import shutil
from IsoNet.util.metadata import MetaData
from IsoNet.util.utils import mkfolder

def run(args):
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    try:

        logging.info('\n######Isonet starts refining######\n')

        if args.continue_from is not None:
            logging.info('\n######Isonet Continues Refining######\n')
            args_continue = load_args_from_json(args.continue_from)
            for item in args_continue.__dict__:
                if args_continue.__dict__[item] is not None and (args.__dict__ is None or not hasattr(args, item)):
                    args.__dict__[item] = args_continue.__dict__[item]
        args = run_whole(args)

        #environment
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if args.log_level == 'debug':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        ### Seperate network with other modules in case we may use pytorch in the future ###
        if True:
            check_gpu(args)
            from IsoNet.models.unet.predict import predict
            from IsoNet.models.unet.train import prepare_first_model, train_data

        ###  find current iterations ###        
        current_iter = args.iter_count if hasattr(args, "iter_count") else 1
        if args.continue_from is not None:
            current_iter += 1

        ###  Main Loop ###
        ###  1. find network model file ###
        ###  2. prediction if network found ###
        ###  3. prepare training data ###
        ###  4. training and save model file ###
        for num_iter in range(current_iter,args.iterations + 1):        
            logging.info("Start Iteration{}!".format(num_iter))

            ### Select a subset of subtomos, useful when the number of subtomo is too large ###
            if args.select_subtomo_number is not None:
                args.mrc_list = np.random.choice(args.all_mrc_list, size = int(args.select_subtomo_number), replace = False)
            else:
                args.mrc_list = args.all_mrc_list

            ### Update the iteration count ###
            args.iter_count = num_iter
            
            if args.pretrained_model is not None:
                ### use pretrained model ###
                mkfolder(args.result_dir)  
                shutil.copyfile(args.pretrained_model,'{}/model_iter{:0>2d}.h5'.format(args.result_dir,num_iter-1))
                logging.info('Use Pretrained model as the output model of iteration {} and predict subtomograms'.format(num_iter-1))
                args.pretrained_model = None
                logging.info("Start predicting subtomograms!")
                predict(args)
                logging.info("Done predicting subtomograms!")
            elif args.continue_from is not None:
                ### Continue from a json file ###
                logging.info('Continue from previous model: {}/model_iter{:0>2d}.h5 of iteration {} and predict subtomograms \
                '.format(args.result_dir,num_iter -1,num_iter-1))
                args.continue_from = None
                logging.info("Start predicting subtomograms!")
                predict(args)
                logging.info("Done predicting subtomograms!")
            elif num_iter == 1:
                ### First iteration ###
                mkfolder(args.result_dir)  
                prepare_first_model(args)
                prepare_first_iter(args)
            else:
                ### Subsequent iterations for all conditions ###
                logging.info("Start predicting subtomograms!")
                predict(args)
                logging.info("Done predicting subtomograms!")

            args.init_model = "{}/model_iter{:0>2d}.h5".format(args.result_dir, num_iter-1)
           
            ### Noise settings ###
            num_noise_volume = 1000
            if num_iter>=args.noise_start_iter[0] and (not os.path.isdir(args.noise_dir) or len(os.listdir(args.noise_dir))< num_noise_volume ):
                from IsoNet.util.noise_generator import make_noise_folder
                
                print(args.noise_mode)
                make_noise_folder(args.noise_dir,args.noise_mode,args.cube_size,num_noise_volume,ncpus=args.preprocessing_ncpus)
            noise_level_series = get_noise_level(args.noise_level,args.noise_start_iter,args.iterations)
            args.noise_level_current =  noise_level_series[num_iter]
            logging.info("Noise Level:{}".format(args.noise_level_current))

            ### remove data_dir and generate training data in data_dir###
            try:
                shutil.rmtree(args.data_dir)     
            except OSError:
                pass
            get_cubes_list(args)
            logging.info("Done preparing subtomograms!")

            ### remove all the mrc files in results_dir ###
            if args.remove_intermediate is True:
                logging.info("Remove intermediate files in iteration {}".format(args.iter_count-1))
                for mrc in args.mrc_list:
                    root_name = mrc.split('/')[-1].split('.')[0]
                    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(args.result_dir,root_name,args.iter_count-1)
                    os.remove(current_mrc)

            ### start training and save model and json ###
            logging.info("Start training!")
            history = train_data(args) #train based on init model and save new one as model_iter{num_iter}.h5
            args.losses = history.history['loss']
            save_args_json(args,args.result_dir+'/refine_iter{:0>2d}.json'.format(num_iter))
            logging.info("Done training!")

            ### for last iteration predict subtomograms ###
            if num_iter == args.iterations and args.remove_intermediate == False:
                logging.info("Predicting subtomograms for last iterations")
                args.iter_count +=1 
                predict(args)
                args.iter_count -=1 

            logging.info("Done Iteration{}!".format(num_iter))

    except Exception:
        import traceback
        error_text = traceback.format_exc()
        f =open('log.txt','a+')
        f.write(error_text)
        f.close()
        logging.error(error_text)
        #logging.error(exc_value)


def run_whole(args):
    '''
    Consume all the argument parameters
    '''
    md = MetaData()
    md.read(args.subtomo_star)
    #*******set fixed parameters*******
    args.crop_size = md._data[0].rlnCropSize
    args.cube_size = md._data[0].rlnCubeSize
    args.predict_cropsize = args.crop_size
    args.residual = True
    #*******calculate parameters********
    if args.gpuID is None:
        args.gpuID = '0,1,2,3'
    else:
        args.gpuID = str(args.gpuID)
    if args.data_dir is None:
        args.data_dir = args.result_dir + '/data'
    if args.iterations is None:
        args.iterations = 30
    args.ngpus = len(list(set(args.gpuID.split(','))))
    if args.result_dir is None:
        args.result_dir = 'results'
    if args.batch_size is None:
        args.batch_size = max(4, 2 * args.ngpus)
    args.predict_batch_size = args.batch_size
    if args.filter_base is None:
        args.filter_base = 64
        # if md._data[0].rlnPixelSize >15:
        #     args.filter_base = 32
        # else:
        #     args.filter_base = 64
    if args.steps_per_epoch is None:
        if args.select_subtomo_number is None:
            args.steps_per_epoch = min(int(len(md) * 6/args.batch_size) , 200)
        else:
            args.steps_per_epoch = min(int(int(args.select_subtomo_number) * 6/args.batch_size) , 200)
    if args.learning_rate is None:
        args.learning_rate = 0.0004
    #if args.noise_level is None:
    #    args.noise_level = (0.05,0.10,0.15,0.20)
    #if args.noise_start_iter is None:
    #    args.noise_start_iter = (11,16,21,26)
    if args.noise_mode is None:
        args.noise_mode = 'noFilter'
    if args.noise_dir is None:
        args.noise_dir = args.result_dir +'/training_noise'
    if args.log_level is None:
        args.log_level = "info"

    if len(md) <=0:
        logging.error("Subtomo list is empty!")
        sys.exit(0)
    args.all_mrc_list = []
    for i,it in enumerate(md):
        if "rlnImageName" in md.getLabels():
            args.all_mrc_list.append(it.rlnImageName)
    return args


def check_gpu(args):
    import subprocess
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_str = out_str[0].decode('utf-8')
    if 'CUDA Version' not in out_str:
        raise RuntimeError('No GPU detected, Please check your CUDA version and installation')

    #import tensorflow related modules after setting environment
    import tensorflow as tf

    gpu_info =  tf.config.list_physical_devices('GPU')
    logging.debug(gpu_info)   
    if len(gpu_info)!=args.ngpus:
        if len(gpu_info) == 0:
            logging.error('No GPU detected, Please check your CUDA version and installation')
            raise RuntimeError('No GPU detected, Please check your CUDA version and installation')
        else:
            logging.error('Available number of GPUs don\'t match requested GPUs \n\n Detected GPUs: {} \n\n Requested GPUs: {}'.format(gpu_info,args.gpuID))
            raise RuntimeError('Re-enter correct gpuID')

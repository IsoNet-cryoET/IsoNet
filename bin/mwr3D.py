from mwr.preprocessing.cubes import prepare_cubes
from mwr.preprocessing.img_processing import normalize
from mwr.preprocessing.prepare import prepare_first_iter,get_cubes_list
#from argparser import args

import mrcfile
import numpy as np
import glob
import logging
import os
import shutil

from mwr.training.train import train_data
from mwr.training.predict import predict



def run(args):
	args.reload_weight = True
	args.result_dir = 'results'
	args.continue_from = "training"
	print('name',__name__)
	# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        #if d_args.log_level == "debug":
	#logger = logging.getLogger(__name__)

	# Specify GPU(s) to be used 
	args.ngpus = len(args.gpuID.split(','))
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID  
	if args.log_level == 'debug':
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	else:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

	if not args.continue_training:
		args.continue_iter = 0
		args = prepare_first_iter(args)
		logging.info("Done preperation for the first iteration!")
	else:
		args.continue_iter = args.continue_iter - 1


	#.mrc2 file should not be in mrc_list 
	mrc_list = glob.glob(args.subtomo_dir+'/*.mrc')
	mrc2_list = glob.glob(args.subtomo_dir+'/*.mrc2') 

	losses = []
	for i in range(args.continue_iter, args.iterations):
		logging.info("Start Iteration{}!".format(i+1))
		args.iter_count = i
		noise_factor = ((args.iter_count - args.noise_start_iter)//args.noise_pause)+1 if args.iter_count > args.noise_start_iter else 0
		logging.info("noise_factor:{}".format(noise_factor))
		if (not args.continue_training) or (args.continue_from == "preprocessing"):
			try:
				shutil.rmtree(args.data_folder)
			except OSError:
				logging.error("Create data folder error!")
			get_cubes_list(args)
			logging.info("Done getting cubes!")
			args.continue_training = False

		if (not args.continue_training) or (args.continue_from == "training"):
			args.mrc_list = os.listdir(args.subtomo_dir)
			args.mrc_list = ['{}/{}'.format(args.subtomo_dir,i) for i in args.mrc_list]
			history = train_data(args)
			losses.append(history.history['loss'][-1])
			args.continue_training = False
			logging.info("Done training!")

		if (not args.continue_training) or (args.continue_from == "predicting"):
			logging.info("Start predicting!")
			args.mrc_list = os.listdir(args.subtomo_dir)
			args.mrc_list = ['{}/{}'.format(args.subtomo_dir,i) for i in args.mrc_list]
			predict(args)
			args.continue_training = False
			logging.info("Done predicting!")

		if len(losses)>3:
			if losses[-1]< losses[-2]:
				logging.warning('loss does not reduce in this iteration')

		logging.info("Done Iteration{}!".format(i+1))
		shutil.rmtree(args.data_folder)
	'''
	losses = []
	for i in range(settings.continue_iter, settings.iterations):
		print('start iteration {}'.format(i+1))
		settings.iter_count = i
		noise_factor = ((settings.iter_count - settings.noise_start_iter)//settings.noise_pause)+1 if settings.iter_count > settings.noise_start_iter else 0
    print('noise_factor',noise_factor)
    if (not settings.continue_training) or (settings.continue_from == "preprocessing"):
        import shutil
        try:
            shutil.rmtree(settings.data_folder)
        except OSError:
            print ("  " )
        get_cubes_list(settings)
        settings.continue_training = False

    if (not settings.continue_training) or (settings.continue_from == "training"):
        history = train_data(settings)
        print(history.history['loss'])
        losses.append(history.history['loss'][-1])
        settings.continue_training = False

    if (not settings.continue_training) or (settings.continue_from == "predicting"):
        predict(settings)
        settings.continue_training = False

    if len(losses)>3:
        if losses[-1]< losses[-2]:
            print('loss does not reduce in this iteration')

'''





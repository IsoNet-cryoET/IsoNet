from mwr.preprocessing.cubes import prepare_cubes
from mwr.preprocessing.img_processing import normalize
from mwr.preprocessing.prepare import prepare_first_iter,get_cubes_list
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
	args.continue_from = "preprocessing"
	args.predict_cropsize = args.crop_size
	args.predict_batch_size = args.batch_size
	args.lr = 0.0004
	# args.filter_base = 64
	if args.log_level == "debug":
		logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
	else:
		logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.INFO)
	logger = logging.getLogger('mwr.preprocessing.prepare')

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
	args.mrc_list = glob.glob(args.subtomo_dir+'/*.mrc')
	args.mrc2_list = glob.glob(args.subtomo_dir+'/*.mrc2')
	args.tomogram2_list = None

	losses = []
	for i in range(args.continue_iter, args.iterations):
		logging.info("Start Iteration{}!".format(i+1))
		args.iter_count = i
		noise_factor = ((args.iter_count - args.noise_start_iter)//args.noise_pause)+1 if args.iter_count > args.noise_start_iter else 0
		logging.info("noise_factor:{}".format(noise_factor))
		if (not args.continue_training) or (args.continue_from == "preprocessing"):
			try:
				shutil.rmtree(args.data_dir)
			except OSError:
				# logging.error("Create data folder error!")
				logging.debug("No previous data folder!")
			#generate training set for next iteration combining original subtomo and subtomo predicted in the last iteration 
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
		shutil.rmtree(args.data_dir)





from preprocessing.cubes import prepare_cubes
from preprocessing.img_processing import normalize
from preprocessing.prepare import prepare_first_iter,get_cubes_list
from argparser import args

import mrcfile
import numpy as np
import glob
import logging
import os
import shutil

from training.train import train_data
from training.predict import predict

if args.log_level == "debug":
	logging.basicConfig(level=logging.DEBUG)

# Specify GPU(s) to be used 
args.ngpus = len(args.gpuID.split(','))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID  


if not args.continue_previous_training:
    args.continue_iter = 0
    args = prepare_first_iter(args)
    logging.info("Done preperation for the first iteration!")


#.mrc2 file should not be in mrc_list 
mrc_list = glob.glob(args.subtomo_dir+'/*.mrc')
mrc2_list = glob.glob(args.subtomo_dir+'/*.mrc2') 
logging.info("Done 2!")

losses = []
for i in range(args.continue_iter, args.iterations):
	#logging.info('start iteration {}'.format(i+1))
	args.iter_count = i
	noise_factor = ((args.iter_count - args.noise_start_iter)//args.noise_pause)+1 if args.iter_count > args.noise_start_iter else 0
	logging.info("noise_factor:{}".format(noise_factor))
	if (not args.continue_previous_training) or (args.continue_from == "preprocessing"):
		try:
			shutil.rmtree(args.data_folder)
		except OSError:
			logging.error("Create data folder error!")
		get_cubes_list(args)
		logging.info("Done getting cubes!")
		args.continue_previous_training = False

	if (not args.continue_previous_training) or (args.continue_from == "training"):
		history = train_data(args)
		logging.info(history.history['loss'])
		losses.append(history.history['loss'][-1])
		args.continue_previous_training = False
		logging.info("Done training!")

	if (not args.continue_previous_training) or (args.continue_from == "predicting"):
		predict(args)
		args.continue_previous_training = False
		logging.info("Done predicting!")

	if len(losses)>3:
		if losses[-1]< losses[-2]:
			logging.warning('loss does not reduce in this iteration')

	logging.info("Done Iteration{}!".format(i+1))

'''
losses = []
for i in range(settings.continue_iter, settings.iterations):
    print('start iteration {}'.format(i+1))
    settings.iter_count = i
    noise_factor = ((settings.iter_count - settings.noise_start_iter)//settings.noise_pause)+1 if settings.iter_count > settings.noise_start_iter else 0
    print('noise_factor',noise_factor)
    if (not settings.continue_previous_training) or (settings.continue_from == "preprocessing"):
        import shutil
        try:
            shutil.rmtree(settings.data_folder)
        except OSError:
            print ("  " )
        get_cubes_list(settings)
        settings.continue_previous_training = False

    if (not settings.continue_previous_training) or (settings.continue_from == "training"):
        history = train_data(settings)
        print(history.history['loss'])
        losses.append(history.history['loss'][-1])
        settings.continue_previous_training = False

    if (not settings.continue_previous_training) or (settings.continue_from == "predicting"):
        predict(settings)
        settings.continue_previous_training = False

    if len(losses)>3:
        if losses[-1]< losses[-2]:
            print('loss does not reduce in this iteration')

'''





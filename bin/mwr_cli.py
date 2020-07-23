#!/usr/bin/env python3
import fire 
from mwr.util.dict2attr import Arg
# from argparser import args
class MWR:
    """
    MWR: Train on tomograms and Predict to restore missing-wedge
    """

    def train(self, input_dir: str,
        gpuID: str = '0,1,2,3',
        datas_are_subtomos = False,
        subtomo_dir='subtomo',
        iterations = 50,
        continue_previous_training = False,
        continue_iter = 0,
        continue_from = "training",
        reload_weight = True,
        preprocessing_ncpus = 28,
        data_folder = "data",
        mask = None,
        result_dir = 'results',
        cube_sidelen = 64,
        cropsize = 96,
        ncube = 32,
        epochs = 1,
        batch_size = 8,
        steps_per_epoch = 200,
        noise_folde = None,
        noise_level = 0.04,
        noise_start_iter = 20,
        noise_pause = 5,
        drop_out = 0.5,
        convs_per_depth = 3,
        kernel = (3,3,3),
        unet_depth = 3,
        batch_normalization = False,
        normalize_percentile = True,
        predict_cropsize = 120,
        predict_batch_size = 8,
        log_level = "debug"):
        """
        Preprocess tomogram and train u-net model on generated subtomos
        :param input_dir: path to tomogram from which subtomos are sampled; format: .mwr or .rec
        :param binv: binvolume of tomogram, will be used to determine the subtomos' size
        :param mask: if sample subtomos with a mask to exclude background region.

        """
        #from mwr.argparser import args
        from mwr.bin.mwr3D import run
        d = locals()
        d_args = Arg(d)
        run(d_args)

    def predict(self, mrc_file: str, output_file: str, weight:str, model: str, gpuID:str='0,1,2,3', cubesize:int=64,cropsize:int=96, batchsize:int=16,norm: bool=True,log_level: str="debug"):
        """
        Predict tomograms using trained model including model.json and weight(xxx.h5)
        :param tomo_path: path to tomogram format: .mwr or .rec
        :param binv: binvolume of tomogram, will be used to determine the subtomos' size
        :param model_path: path to trained model
        :param output: 
        """
        from mwr.bin.mwr3D_predict import predict
        d = locals()
        d_args = Arg(d)
        predict(d_args)

    def make_mask(self,tomo_path,mask_name,side=8,percentile=50,threshold=1):
        """
        generate a mask to constrain sampling area of the tomogram
        :param tomo_path: path to the tomogram
        :param mask_name: path and name of the mask to save as
        :param side:
        :param percentile:
        :param threshold:
        """
        from mwr.bin.maskGene import make_mask
        make_mask(tomo_path,mask_name,side=side,percentile=percentile,threshold=threshold)
        print('mask generated')

    def generate_noise(self,output_folder: str,number_volume: int, cubesize: int, minangle: int=-60,maxangle: int=60,
    anglestep: int=2, start: int=0,ncpus: int=20, mode: int=0):
        """
        generate training noise to accelerate the missing wedge information retrieval
        :param output_folder: path to folder for saving noises
        :param number_volume: number of noise cubes to generate
        :param cubesize: side length of the noise cubes, usually 64 or 96
        :param ncpus: number of cpus to use
        """
        from mwr.util.mwr3D_noise_generator import make_noise
        make_noise(output_folder=output_folder, number_volume=number_volume, cubesize=cubesize, minangle=minangle,maxangle=maxangle, anglestep=anglestep, start=start,ncpus=ncpus, mode=mode)
        
    
    
if __name__ == "__main__":
    fire.Fire(MWR)

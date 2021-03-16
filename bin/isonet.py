#!/usr/bin/env python3
import fire
import logging
import os
from IsoNet.util.dict2attr import Arg,check_args
from IsoNet.util.deconvolution import tom_deconv_tomo
import sys
from IsoNet.preprocessing.cubes import mask_mesh_seeds
from fire import core
class ISONET:
    """
    ISONET: Train on tomograms and Predict to restore missing-wedge
    """
    def refine(self,
        input_dir: str = None,
        gpuID: str = '0,1,2,3',
        mask_dir: str= None,
        noise_dir: str = None,
        iterations: int = 50,
        data_dir: str = "data",
        pretrained_model = None,
        log_level: str = "info",
        continue_iter: int = 0,

        noise_mode: int=1,
        noise_level:  float= 0.05,
        noise_start_iter: int = 15,
        noise_pause: int = 5,

        cube_size: int = 64,
        crop_size: int = 96,
        ncube: int = 1,
        preprocessing_ncpus: int = 16,

        epochs: int = 10,
        batch_size: int = 8,
        steps_per_epoch: int = 150,

        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = 32,
        batch_normalization: bool = False,
        normalize_percentile: bool = True,
    ):
        """
        Preprocess tomogram and train u-net model on generated subtomos
        :param input_dir: path to tomogram from which subtomos are sampled; format: .mrc or .rec
        :param mask: (None) folder of  mask files
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param datas_are_subtomos: (False) Is your trainin data subtomograms?
        :param subtomo_dir: (subtomo) The folder where the input subtomograms at.
        :param iterations: (50) Number of training iterations.
        :param continue_training: (False) Continus previous training? When continue, the architecture of network can not be changed.
        :param continue_iter: (0) Which iteration you want to start from?

        ************************noise settings************************

        :param log_level: (debug) logging level

        ************************continue training settings************************

        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************training settings************************

        :param data_folder: (data)Temperary folder to save the generated data used for training.
        :param cube_sidelen: Size of training cubes, this size should be divisible by 2^unet_depth.
        :param cropsize: Size of cubes to impose missing wedge. Should be same or larger than size of cubes.
        :param ncube: Number of cubes generated for each tomogram. Because each sampled subtomogram rotates 16 times, the actual number of subtomograms for trainings should be ncube*16.
        :param epochs: Number of epoch for each iteraction.
        :param batch_size:Size of the minibatch.
        :param steps_per_epoch:Step per epoch. A good estimation of this value is tomograms * ncube * 16 / batch_size *0.9.")

        ************************network settings************************

        :param noise_folder: Add noise during training, Set None to disable noise reduction.
        :param noise_level: Level of noise STD(added noise)/STD(data).
        :param noise_start_iter: Iteration that start to add trainning noise.
        :param noise_pause: Iters trainning noise remain at one level.

        ************************preprocessing settings************************

        :param drop_out: Drop out rate to reduce overfitting.
        :param convs_per_depth: Number of convolution layer for each depth.
        :param kernel: Kernel for convolution
        :param unet_depth: Number of convolution layer for each depth.
        :param filter_base: The base number of channels after convolution
        :param batch_normalization: Sometimes batch normalization may induce artifacts for extreme pixels in the first several iterations. Those could be restored in further iterations.
        :param normalize_percentile:Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.

        Typical training strategy:
        1. Train tomo with no pretrained model
        2. Continue train with previous interupted model
        3. Continue train with pre-trained model
        """

        from IsoNet.bin.refine import run

        d = locals()
        d_args = Arg(d)
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.INFO)
        # logging.basicConfig(level=logging.WARNING,format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        # datefmt='%Y-%m-%d:%H:%M:%S')
        #if d_args.log_level == "debug":
        # logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('IsoNet.bin.refine')
        d_args.only_extract_subtomos = False
        run(d_args)

    def predict(self, mrc_file: str, output_file: str, model: str, gpuID: str = None, cube_size:int=64,crop_size:int=96, batch_size:int=8,norm: bool=True,log_level: str="debug"):
        """
        Predict tomograms using trained model including model.json and weight(xxx.h5)
        :param mrc_file: path to tomogram, format: .mrc or .rec
        :param output_file: file_name of predicted tomograms
        :param model: path to trained model
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping strategy
        :param batch_size: The batch size of the cubes grouped into for network predicting
        :param norm: (True) if normalize the tomograms by percentile
        :param log_level: ("debug") level of message to be displayed
        :raises: AttributeError, KeyError
        """
        from IsoNet.bin.predict import predict
        d = locals()

        d_args = Arg(d)
        predict(d_args)

    def make_mask(self,tomo_path,mask_path: str = None,side: int=8,percentile: int=99,threshold: int=1,mask_type: str="statistical"):
        """
        generate a mask to constrain sampling area of the tomogram
        :param tomo_path: path to the tomogram or tomogram folder
        :param mask_path: path and name of the mask to save as
        :param side:
        :param percentile:
        :param threshold:
        :param mask_type: 'statistical' or 'surface': Masks can be generated based on the statistics or just take the middle part of tomograms
        """
        from IsoNet.bin.make_mask import make_mask,make_mask_dir
        if os.path.isdir(tomo_path):
            make_mask_dir(tomo_path,mask_path,side=side,percentile=percentile,threshold=threshold,mask_type=mask_type)
        elif os.path.isfile(tomo_path):
            if mask_path is None:
                mask_path = tomo_path.split('.')[0]+'_mask.mrc'
                print(mask_path)
            make_mask(tomo_path,mask_path,side=side,percentile=percentile,threshold=threshold,mask_type=mask_type)
        else:
            print('make_mask tomo_path error')
        print('mask generated')

    def generate_noise(self,output_folder: str,number_volume: int, cubesize: int, minangle: int=-60,maxangle: int=60,
    anglestep: int=2, start: int=0,ncpus: int=20, mode: int=1):
        """
        Generate training noise to accelerate the missing wedge information retrieval. This commond will generate a folder containing noise volumes which mimics the distorded noise pattern in the tomograms with size of cubesize x cubesize x cubesize. The noise volumes are indexed from start to start + number_volume
        :param output_folder: path to folder for saving noises
        :param number_volume: number of noise cubes to generate
        :param cubesize: side length of the noise cubes, usually 64 or 96
        :param ncpus: number of cpus to use
        :param minangle: the minimal angle of your tilt series
        :param maxangle: the maximal angle of your tilt series
        :param anglestep: the step of your tilt series' angles
        :param start: When you want to add additional noise volumes, you can specify the start value as the number of already generated noise volumes. So the alreaded generated volumes will not be ovewrited.
        :param mode: mode=1, noise is reconstructed by back-projection algorithm; mode=2 or else, noise is gained by filtering gaussian noise volumes.
        """
        from IsoNet.util.noise_generator import make_noise
        make_noise(output_folder=output_folder, number_volume=number_volume, cubesize=cubesize, minangle=minangle,maxangle=maxangle, anglestep=anglestep, start=start,ncpus=ncpus, mode=mode)

    def check(self):
        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        print('IsoNet --version 0.9.9 installed')

    def generate_command(self, tomo_dir: str, mask_dir: str=None, ncpu: int=10, gpu_memory: int=10, ngpu: int=4, pixel_size: float=10, also_denoise: bool=True):
        """
        G
        :param pixel_size: pixel size in anstrom
        """
        import mrcfile
        import numpy as np
        s="isonet.py refine --input_dir {} ".format(tomo_dir)
        if mask_dir is not None:
            s+="--mask_dir {} ".format(mask_dir)
            m=os.listdir(mask_dir)
            with mrcfile.open(mask_dir+"/"+m[0]) as mrcData:
                mask_data = mrcData.data
            # vsize=np.count_nonzero(mask_data)
        else:
            m=os.listdir(tomo_dir)
            with mrcfile.open(tomo_dir+"/"+m[0]) as mrcData:
                tomo_data = mrcData.data
            sh=tomo_data.shape
            mask_data = np.ones(sh)
        num_tomo = len(m)

        s+="--preprocessing_ncpus {} ".format(ncpu)
        s+="--gpuID "
        for i in range(ngpu-1):
            s+=str(i)
            s+=","
        s+=str(ngpu-1)
        s+=" "
        if pixel_size < 15.0:
            filter_base = 64
            s+="--filter_base 64 "
        else:
            filter_base = 32
            s+="--filter_base 32"
        if ngpu < 6:
            batch_size = 2 * ngpu
            s+="--batch_size {} ".format(batch_size)
        # elif ngpu == 3:
        #     batch_size = 6
        #     s+="--batch_size 6 "
        else:
            batch_size = ngpu
            s+="--batch_size {} ".format(ngpu)
        if filter_base==64:
            cube_size = int((gpu_memory/(batch_size/ngpu)) ** (1/3.0) *40 /16)*16
        elif filter_base ==32:
            cube_size = int((gpu_memory*3/(batch_size/ngpu)) ** (1/3.0) *40 /16)*16

        if cube_size == 0:
            print("Please use larger memory GPU or use more GPUs")

        s+="--cube_size {} --crop_size {} ".format(cube_size, int(cube_size*1.5))

        # num_per_tomo = int(vsize/(cube_size**3) * 0.5)
        num_per_tomo = len(mask_mesh_seeds(mask_data,cube_size)[0] )
        s+="--ncube {} ".format(num_per_tomo)

        num_particles = int(num_per_tomo * num_tomo * 16 * 0.9)
        s+="--epochs 10 --steps_per_epoch {} ".format(int(num_particles/batch_size*0.4))

        if also_denoise:
            s+="--iterations 40 --noise_level 0.05 --noise_start_iter 15 --noise_pause 3"
        else:
            s+="--iterations 15 --noise_level 0 --noise_start_iter 100"
        print(s)

    def deconv(self,tomo, defocus: float=1.0, pixel_size: float=1.0,snrfalloff: float=1.0, deconvstrength: float=1.0):
        import mrcfile
        with mrcfile.open(tomo) as mrc:
            vol = mrc.data
        result = tom_deconv_tomo(vol, angpix=pixel_size, defocus=defocus, snrfalloff=snrfalloff, deconvstrength=deconvstrength, highpassnyquist=0.1, phaseflipped=False, phaseshift=0 )
        outname = tomo.split('.')[0] +'-deconv.rec'
        with mrcfile.new(outname, overwrite=True) as mrc:
            mrc.set_data(result)

    def extract(self,
        input_dir: str = None,
        mask_dir: str= None,
        crop_size: int = 96,
        ncube: int = 1,
    ):
        from IsoNet.preprocessing.prepare import extract_subtomos
        d = locals()
        d_args = Arg(d)
        d_args.only_extract_subtomos = True
        extract_subtomos(d_args)

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

if __name__ == "__main__":
    core.Display = Display
    fire.Fire(ISONET)

#!/usr/bin/env python3
import fire
import logging
import os, sys, traceback
from IsoNet.util.dict2attr import Arg,check_parse,idx2list
from fire import core
from IsoNet.util.metadata import MetaData,Label,Item

class ISONET:
    """
    ISONET: Train on tomograms and restore missing-wedge\n
    for detail discription, run one of the following commands:

    isonet.py prepare_star -h
    isonet.py prepare_subtomo_star -h
    isonet.py deconv -h
    isonet.py make_mask -h
    isonet.py extract -h
    isonet.py refine -h
    isonet.py predict -h
    isonet.py resize -h
    isonet.py gui -h
    """
    #log_file = "log.txt"

    def prepare_star(self,folder_name, output_star='tomograms.star',pixel_size = 10.0, defocus = 0.0, number_subtomos = 100):
        """
        \nThis command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).\n
        isonet.py prepare_star folder_name [--output_star] [--pixel_size] [--defocus] [--number_subtomos]
        :param folder_name: (None) directory containing tomogram(s). Usually 1-5 tomograms are sufficient.
        :param output_star: (tomograms.star) star file similar to that from "relion". You can modify this file manually or with gui.
        :param pixel_size: (10) pixel size in anstroms. Usually you want to bin your tomograms to about 10A pixel size.
        Too large or too small pixel sizes are not recommanded, since the target resolution on Z-axis of corrected tomograms should be about 30A.
        :param defocus: (0.0) defocus in Angstrom. Only need for ctf deconvolution. For phase plate data, you can leave defocus 0.
        If you have multiple tomograms with different defocus, please modify them in star file or with gui.
        :param number_subtomos: (100) Number of subtomograms to be extracted in later processes.
        If you want to extract different number of subtomograms in different tomograms, you can modify them in the star file generated with this command or with gui.

        """
        md = MetaData()
        md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo','rlnMaskBoundary')
        tomo_list = sorted(os.listdir(folder_name))
        i = 0
        for tomo in tomo_list:
            if tomo[-4:] == '.rec' or tomo[-4:] == '.mrc':
                i+=1
                it = Item()
                md.addItem(it)
                md._setItemValue(it,Label('rlnIndex'),str(i))
                md._setItemValue(it,Label('rlnMicrographName'),os.path.join(folder_name,tomo))
                md._setItemValue(it,Label('rlnPixelSize'),pixel_size)
                md._setItemValue(it,Label('rlnDefocus'),defocus)
                md._setItemValue(it,Label('rlnNumberSubtomo'),number_subtomos)
                md._setItemValue(it,Label('rlnMaskBoundary'),None)
        md.write(output_star)

    def prepare_subtomo_star(self, folder_name, output_star='subtomo.star', pixel_size: float=10.0, cube_size = None):
        """
        \nThis command generates a subtomo star file from a folder containing only subtomogram files (.mrc).
        This command is usually not necessary in the traditional workflow, because "isonet.py extract" will generate this subtomo.star for you.\n
        isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size]
        :param folder_name: (None) directory containing subtomogram(s).
        :param output_star: (subtomo.star) output star file for subtomograms, will be used as input in refinement.
        :param pixel_size: (10) The pixel size in angstrom of your subtomograms.
        :param cube_size: (None) This is the size of the cubic volumes used for training. This values should be smaller than the size of subtomogram.
        And the cube_size should be divisible by 8. If this value isn't set, cube_size is automatically determined as int(subtomo_size / 1.5 + 1)//16 * 16
        """
        #TODO check folder valid, logging
        if not os.path.isdir(folder_name):
            print("the folder does not exist")
        import mrcfile
        md = MetaData()
        md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
        subtomo_list = sorted(os.listdir(folder_name))
        for i,subtomo in enumerate(subtomo_list):
            subtomo_name = os.path.join(folder_name,subtomo)
            try:
                with mrcfile.open(subtomo_name, mode='r') as s:
                    crop_size = s.header.nx
            except:
                print("Warning: Can not process the subtomogram: {}!".format(subtomo_name))
                continue
            if cube_size is not None:
                cube_size = int(cube_size)
                if cube_size >= crop_size:
                    cube_size = int(crop_size / 1.5 + 1)//16 * 16
                    print("Warning: Cube size should be smaller than the size of subtomogram volume! Using cube size {}!".format(cube_size))
            else:
                cube_size = int(crop_size / 1.5 + 1)//16 * 16
            it = Item()
            md.addItem(it)
            md._setItemValue(it,Label('rlnSubtomoIndex'),str(i+1))
            md._setItemValue(it,Label('rlnImageName'),subtomo_name)
            md._setItemValue(it,Label('rlnCubeSize'),cube_size)
            md._setItemValue(it,Label('rlnCropSize'),crop_size)
            md._setItemValue(it,Label('rlnPixelSize'),pixel_size)

            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        md.write(output_star)

    def deconv(self, star_file: str,
        deconv_folder:str="./deconv",
        snrfalloff: float=None,
        deconvstrength: float=None,
        highpassnyquist: float=0.02,
        chunk_size: int=None,
        overlap_rate: float= 0.25,
        ncpu:int=4,
        tomo_idx: str=None):
        """
        \nCTF deconvolution for the tomograms.\n
        isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--overlap_rate] [--ncpu] [--tomo_idx]
        This step is recommanded because it enhances low resolution information for a better contrast. No need to do deconvolution for phase plate data.
        :param deconv_folder: (./deconv) Folder created to save deconvoluted tomograms.
        :param star_file: (None) Star file for tomograms.
        :param snrfalloff: (1.0) SNR fall rate with the frequency. High values means losing more high frequency.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param deconvstrength: (1.0) Strength of the deconvolution.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param highpassnyquist: (0.02) Highpass filter for at very low frequency. We suggest to keep this default value.
        :param chunk_size: (None) When your computer has enough memory, please keep the chunk_size as the default value: None . Otherwise, you can let the program crop the tomogram into multiple chunks for multiprocessing and assembly them into one. The chunk_size defines the size of individual chunk. This option may induce artifacts along edges of chunks. When that happen, you may use larger overlap_rate.
        :param overlap_rate: (None) The overlapping rate for adjecent chunks.
        :param ncpu: (4) Number of cpus to use.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        """
        from IsoNet.util.deconvolution import deconv_one

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        logging.info('\n######Isonet starts ctf deconvolve######\n')

        try:
            md = MetaData()
            md.read(star_file)
            if not 'rlnSnrFalloff' in md.getLabels():
                md.addLabels('rlnSnrFalloff','rlnDeconvStrength','rlnDeconvTomoName')
                for it in md:
                    md._setItemValue(it,Label('rlnSnrFalloff'),1.0)
                    md._setItemValue(it,Label('rlnDeconvStrength'),1.0)
                    md._setItemValue(it,Label('rlnDeconvTomoName'),None)

            if not os.path.isdir(deconv_folder):
                os.mkdir(deconv_folder)

            tomo_idx = idx2list(tomo_idx)
            for it in md:
                if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                    if snrfalloff is not None:
                        md._setItemValue(it,Label('rlnSnrFalloff'), snrfalloff)
                    if deconvstrength is not None:
                        md._setItemValue(it,Label('rlnDeconvStrength'),deconvstrength)

                    tomo_file = it.rlnMicrographName
                    base_name = os.path.basename(tomo_file)
                    deconv_tomo_name = '{}/{}'.format(deconv_folder,base_name)

                    deconv_one(it.rlnMicrographName,deconv_tomo_name,defocus=it.rlnDefocus/10000.0, pixel_size=it.rlnPixelSize,snrfalloff=it.rlnSnrFalloff, deconvstrength=it.rlnDeconvStrength,highpassnyquist=highpassnyquist,chunk_size=chunk_size,overlap_rate=overlap_rate,ncpu=ncpu)
                    md._setItemValue(it,Label('rlnDeconvTomoName'),deconv_tomo_name)
                md.write(star_file)
            logging.info('\n######Isonet done ctf deconvolve######\n')

        except Exception:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    def make_mask(self,star_file,
                mask_folder: str = 'mask',
                patch_size: int=4,
                mask_boundary: str=None,
                density_percentage: int=None,
                std_percentage: int=None,
                use_deconv_tomo:bool=True,
                z_crop:float=None,
                tomo_idx=None):
        """
        \ngenerate a mask that include sample area and exclude "empty" area of the tomogram. The masks do not need to be precise. In general, the number of subtomograms (a value in star file) should be lesser if you masked out larger area. \n
        isonet.py make_mask star_file [--mask_folder] [--patch_size] [--density_percentage] [--std_percentage] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: path to the tomogram or tomogram folder
        :param mask_folder: path and name of the mask to save as
        :param patch_size: (4) The size of the box from which the max-filter and std-filter are calculated.
        :param density_percentage: (50) The approximate percentage of pixels to keep based on their local pixel density.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param std_percentage: (50) The approximate percentage of pixels to keep based on their local standard deviation.
        If this value is not set, the program will look for the parameter in the star file.
        If this value is not set and not found in star file, the default value 50 will be used.
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :param z_crop: If exclude the top and bottom regions of tomograms along z axis. For example, "--z_crop 0.2" will mask out the top 20% and bottom 20% region along z axis.
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        """
        from IsoNet.bin.make_mask import make_mask
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        logging.info('\n######Isonet starts making mask######\n')
        try:
            if not os.path.isdir(mask_folder):
                os.mkdir(mask_folder)
            # write star percentile threshold
            md = MetaData()
            md.read(star_file)
            if not 'rlnMaskDensityPercentage' in md.getLabels():
                md.addLabels('rlnMaskDensityPercentage','rlnMaskStdPercentage','rlnMaskName')
                for it in md:
                    md._setItemValue(it,Label('rlnMaskDensityPercentage'),50)
                    md._setItemValue(it,Label('rlnMaskStdPercentage'),50)
                    md._setItemValue(it,Label('rlnMaskName'),None)

            tomo_idx = idx2list(tomo_idx)
            for it in md:
                if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                    if density_percentage is not None:
                        md._setItemValue(it,Label('rlnMaskDensityPercentage'),density_percentage)
                    if std_percentage is not None:
                        md._setItemValue(it,Label('rlnMaskStdPercentage'),std_percentage)
                    if use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and it.rlnDeconvTomoName not in [None,'None']:
                        tomo_file = it.rlnDeconvTomoName
                    else:
                        tomo_file = it.rlnMicrographName
                    tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]

                    if os.path.isfile(tomo_file):
                        logging.info('make_mask: {}| dir_to_save: {}| percentage: {}| window_scale: {}'.format(tomo_file,
                        mask_folder, it.rlnMaskDensityPercentage, patch_size))
                        
                        #if mask_boundary is None:
                        if "rlnMaskBoundary" in md.getLabels() and it.rlnMaskBoundary not in [None, "None"]:
                            mask_boundary = it.rlnMaskBoundary 
                        else:
                            mask_boundary = None
                              
                        mask_out_name = '{}/{}_mask.mrc'.format(mask_folder,tomo_root_name)
                        make_mask(tomo_file,
                                mask_out_name,
                                mask_boundary=mask_boundary,
                                side=patch_size,
                                density_percentage=it.rlnMaskDensityPercentage,
                                std_percentage=it.rlnMaskStdPercentage,
                                surface = z_crop)

                    md._setItemValue(it,Label('rlnMaskName'),mask_out_name)
                md.write(star_file)
            logging.info('\n######Isonet done making mask######\n')
        except Exception:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)

    def extract(self,
        star_file: str,
        use_deconv_tomo: bool = True,
        subtomo_folder: str = "subtomo",
        subtomo_star: str = "subtomo.star",
        cube_size: int = 64,
        crop_size: int = None,
        log_level: str="info",
        tomo_idx = None
        ):

        """
        \nExtract subtomograms\n
        isonet.py extract star_file [--subtomo_folder] [--subtomo_star] [--cube_size] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: tomogram star file
        :param subtomo_folder: (subtomo) folder for output subtomograms.
        :param subtomo_star: (subtomo.star) star file for output subtomograms.
        :param cube_size: (64) Size of cubes for training, should be divisible by 8, eg. 32, 64. The actual sizes of extracted subtomograms are this value adds 16.
        :param crop_size: (None) The size of subtomogram, should be larger then the cube_size The default value is 16+cube_size.
        :param log_level: ("info") level of the output, either "info" or "debug"
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        """
        d = locals()
        d_args = Arg(d)

        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

        logging.info("\n######Isonet starts extracting subtomograms######\n")

        try:
            if os.path.isdir(subtomo_folder):
                logging.warning("subtomo directory exists, the current directory will be overwriten")
                import shutil
                shutil.rmtree(subtomo_folder)
            os.mkdir(subtomo_folder)

            from IsoNet.preprocessing.prepare import extract_subtomos
            if crop_size is None:
                d_args.crop_size = cube_size + 16
            else:
                d_args.crop_size = crop_size
            d_args.subtomo_dir = subtomo_folder
            d_args.tomo_idx = idx2list(tomo_idx)
            extract_subtomos(d_args)
            logging.info("\n######Isonet done extracting subtomograms######\n")
        except Exception:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)


    def refine(self,
        subtomo_star: str,
        gpuID: str = None,
        iterations: int = None,
        data_dir: str = None,
        pretrained_model: str = None,
        log_level: str = None,
        result_dir: str='results',
        remove_intermediate: bool =False,
        select_subtomo_number: int = None,
        preprocessing_ncpus: int = 16,
        continue_from: str=None,
        epochs: int = 10,
        batch_size: int = None,
        steps_per_epoch: int = None,

        noise_level:  tuple=(0.05,0.10,0.15,0.20),
        noise_start_iter: tuple=(11,16,21,26),
        noise_mode: str = None,
        noise_dir: str = None,
        learning_rate: float = None,
        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = None,
        batch_normalization: bool = True,
        normalize_percentile: bool = True,

    ):
        """
        \ntrain neural network to correct missing wedge\n
        isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level]...
        :param subtomo_star: (None) star file containing subtomogram(s).
        :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
        :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
        :param iterations: (30) Number of training iterations.
        :param data_dir: (data) Temperary folder to save the generated data used for training.
        :param log_level: (info) debug level, could be 'info' or 'debug'
        :param continue_from: (None) A Json file to continue from. That json file is generated at each iteration of refine.
        :param result_dir: ('results') The name of directory to save refined neural network models and subtomograms
        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************Training settings************************

        :param epochs: (10) Number of epoch for each iteraction.
        :param batch_size: (None) Size of the minibatch.If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param steps_per_epoch: (None) Step per epoch. If not defined, the default value will be min(num_of_subtomograms * 6 / batch_size , 200)

        ************************Denoise settings************************

        :param noise_level: (0.05,0.1,0.15,0.2) Level of noise STD(added noise)/STD(data) after the iteration defined in noise_start_iter.
        :param noise_start_iter: (11,16,21,26) Iteration that start to add noise of corresponding noise level.
        :param noise_mode: (None) Filter names when generating noise volumes, can be 'ramp', 'hamming' and 'noFilter'
        :param noise_dir: (None) Directory for generated noise volumes. If set to None, the Noise volumes should appear in results/training_noise

        ************************Network settings************************

        :param drop_out: (0.3) Drop out rate to reduce overfitting.
        :param learning_rate: (0.0004) learning rate for network training.
        :param convs_per_depth: (3) Number of convolution layer for each depth.
        :param kernel: (3,3,3) Kernel for convolution
        :param unet_depth: (3) Depth of UNet.
        :param filter_base: (64) The base number of channels after convolution.
        :param batch_normalization: (True) Use Batch Normalization layer
        :param pool: (False) Use pooling layer instead of stride convolution layer.
        :param normalize_percentile: (True) Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.
        """
        from IsoNet.bin.refine import run
        d = locals()
        d_args = Arg(d)
        with open('log.txt','a+') as f:
            f.write(' '.join(sys.argv[0:]) + '\n')
        run(d_args)

    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=64,
    crop_size:int=96,use_deconv_tomo=True, batch_size:int=None,normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
        """
        \nPredict tomograms using trained model\n
        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
        :param star_file: star for tomograms.
        :param output_dir: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
        :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
        :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
        :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
        :raises: AttributeError, KeyError
        """
        d = locals()
        d_args = Arg(d)
        from IsoNet.bin.predict import predict

        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
            datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        try:
            predict(d_args)
        except:
            error_text = traceback.format_exc()
            f =open('log.txt','a+')
            f.write(error_text)
            f.close()
            logging.error(error_text)
    
    def resize(self, star_file:str, apix: float=15, out_folder="tomograms_resized"):
        '''
        This function rescale the tomograms to a given pixelsize
        '''
        md = MetaData()
        md.read(star_file)
        #print(md._data[0].rlnPixelSize)
        from scipy.ndimage import zoom
        #from skimage.transform import rescale
        #import numpy as np
        import mrcfile
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        for item in md._data:
            ori_apix = item.rlnPixelSize
            tomo_name = item.rlnMicrographName
            zoom_factor = float(ori_apix)/apix
            new_tomo_name = "{}/{}".format(out_folder,os.path.basename(tomo_name))
            with mrcfile.open(tomo_name) as mrc:
                data = mrc.data
            print("scaling: {}".format(tomo_name))
            new_data = zoom(data, zoom_factor,order=3, prefilter=False)
            #new_data = rescale(data, zoom_factor,order=3, anti_aliasing = True)
            #new_data = new_data.astype(np.float32)

            with mrcfile.new(new_tomo_name,overwrite=True) as mrc:
                mrc.set_data(new_data)
                mrc.voxel_size = apix

            item.rlnPixelSize = apix
            print(new_tomo_name)
            item.rlnMicrographName = new_tomo_name
            print(item.rlnMicrographName)
        md.write(os.path.splitext(star_file)[0] + "_resized.star")
        print("scale_finished")

    def check(self):
        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        import skimage
        import PyQt5
        import tqdm
        print('IsoNet --version 0.2 installed')

    def gui(self):
        """
        \nGraphic User Interface\n
        """
        import IsoNet.gui.Isonet_star_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

def pool_process(p_func,chunks_list,ncpu):
    from multiprocessing import Pool
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results

if __name__ == "__main__":
    core.Display = Display
    # logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
        check_parse(sys.argv[1:])
    fire.Fire(ISONET)

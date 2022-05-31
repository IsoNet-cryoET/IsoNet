import os 
import sys
import logging
import sys
import mrcfile
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.preprocessing.simulate import apply_wedge1 as  apply_wedge, mw2d
from IsoNet.preprocessing.simulate import apply_wedge_dcube
from multiprocessing import Pool
import numpy as np
from functools import partial
from IsoNet.util.rotations import rotation_list
# from difflib import get_close_matches
from IsoNet.util.metadata import MetaData, Item, Label
#Make a new folder. If exist, nenew it
# Do not set basic config for logging here
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)

def extract_subtomos(settings):
    '''
    extract subtomo from whole tomogram based on mask
    and feed to generate_first_iter_mrc to generate xx_iter00.xx
    '''
    md = MetaData()
    md.read(settings.star_file)
    if len(md)==0:
        sys.exit("No input exists. Please check it in input folder!")

    subtomo_md = MetaData()
    subtomo_md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
    count=0
    for it in md:
        if settings.tomo_idx is None or str(it.rlnIndex) in settings.tomo_idx:
            pixel_size = it.rlnPixelSize
            if settings.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and os.path.isfile(it.rlnDeconvTomoName):
                logging.info("Extract from deconvolved tomogram {}".format(it.rlnDeconvTomoName))
                with mrcfile.open(it.rlnDeconvTomoName) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            else:        
                print("Extract from origional tomogram {}".format(it.rlnMicrographName))
                with mrcfile.open(it.rlnMicrographName) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)
            

            if "rlnMaskName" in md.getLabels() and it.rlnMaskName not in [None, "None"]:
                with mrcfile.open(it.rlnMaskName) as m:
                    mask_data = m.data
            else:
                mask_data = None
                logging.info(" mask not been used for tomogram {}!".format(it.rlnIndex))

            seeds=create_cube_seeds(orig_data, it.rlnNumberSubtomo, settings.crop_size,mask=mask_data)
            subtomos=crop_cubes(orig_data,seeds,settings.crop_size)

            # save sampled subtomo to {results_dir}/subtomos instead of subtomo_dir (as previously does)
            base_name = os.path.splitext(os.path.basename(it.rlnMicrographName))[0]
            
            for j,s in enumerate(subtomos):
                im_name = '{}/{}_{:0>6d}.mrc'.format(settings.subtomo_dir, base_name, j)
                with mrcfile.new(im_name, overwrite=True) as output_mrc:
                    count+=1
                    subtomo_it = Item()
                    subtomo_md.addItem(subtomo_it)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnSubtomoIndex'), str(count))
                    subtomo_md._setItemValue(subtomo_it,Label('rlnImageName'), im_name)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCubeSize'),settings.cube_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnCropSize'),settings.crop_size)
                    subtomo_md._setItemValue(subtomo_it,Label('rlnPixelSize'),pixel_size)
                    output_mrc.set_data(s.astype(np.float32))
    subtomo_md.write(settings.subtomo_star)

def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]

def get_cubes_one(data_X, data_Y, settings, start = 0, mask = None, add_noise = 0):
    '''
    crop out one subtomo and missing wedge simulated one from input data,
    and save them as train set
    '''
    #data_X = apply_wedge_dcube(data, mw)
    #data_Y = crop_to_size(data, settings.crop_size, settings.cube_size)
    #data_X = crop_to_size(apply_wedge_dcube(data, mw), settings.crop_size, settings.cube_size)

    if settings.noise_level_current > 0.0000001:
        if settings.noise_dir is not None:
            path_noise = sorted([settings.noise_dir+'/'+f for f in os.listdir(settings.noise_dir)])
            path_index = np.random.randint(len(path_noise))
            def read_vol(f):
                with mrcfile.open(f) as mf:
                    res = mf.data
                return res
            noise_volume = read_vol(path_noise[path_index])
        else:
            from IsoNet.util.noise_generator import make_noise_one
            noise_volume = make_noise_one(cubesize = settings.cube_size,mode=settings.noise_mode)
        
        #Noise along y axis is indenpedent, so that the y axis can be permutated.
        noise_volume = np.transpose(noise_volume, axes=(1,0,2))
        noise_volume = np.random.permutation(noise_volume)
        noise_volume = np.transpose(noise_volume, axes=(1,0,2))
        data_X += settings.noise_level_current * noise_volume / np.std(noise_volume)

    with mrcfile.new('{}/train_x/x_{}.mrc'.format(settings.data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_X.astype(np.float32))
    with mrcfile.new('{}/train_y/y_{}.mrc'.format(settings.data_dir, start), overwrite=True) as output_mrc:
        output_mrc.set_data(data_Y.astype(np.float32))
    return 0


def get_cubes(inp,settings):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]
    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count-1)
    with mrcfile.open(mrc) as mrcData:
        iw_data = mrcData.data.astype(np.float32)*-1
    iw_data = normalize(iw_data, percentile = settings.normalize_percentile)

    with mrcfile.open(current_mrc) as mrcData:
        ow_data = mrcData.data.astype(np.float32)*-1
    ow_data = normalize(ow_data, percentile = settings.normalize_percentile)

    orig_data = apply_wedge(ow_data, ld1=0, ld2=1) + apply_wedge(iw_data, ld1 = 1, ld2=0)
    #orig_data = ow_data
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape))

    old_rotation = True
    if old_rotation:
        for i,r in enumerate(rotation_list):
            data = np.rot90(orig_data, k=r[0][1], axes=r[0][0])
            data = np.rot90(data, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group 
        for i in range(len(rotation_list)):
            rot = special_ortho_group.rvs(3)
            center = (np.array(orig_data.shape) -1 )/2.
            offset = center-np.dot(rot,center)
            rotated_data[i] = affine_transform(orig_data,rot,offset=offset)
    
    mw = mw2d(settings.crop_size)   
    datax = apply_wedge_dcube(rotated_data, mw)

    for i in range(len(rotation_list)): 
        data_X = crop_to_size(datax[i], settings.crop_size, settings.cube_size)
        data_Y = crop_to_size(rotated_data[i], settings.crop_size, settings.cube_size)
        get_cubes_one(data_X, data_Y, settings, start = start) 
        start += 1#settings.ncube

def get_cubes_list(settings):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    if not os.path.exists(settings.data_dir):
        os.makedirs(settings.data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(settings.data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)
    inp=[]
    for i,mrc in enumerate(settings.mrc_list):
        inp.append((mrc, i*len(rotation_list)))
    
    # inp: list 0f (mrc_dir, index * rotation times)

    if settings.preprocessing_ncpus > 1:
        func = partial(get_cubes, settings=settings)
        with Pool(settings.preprocessing_ncpus) as p:
            p.map(func,inp)
    else:
        for i in inp:
            logging.info("{}".format(i))
            get_cubes(i, settings)

    all_path_x = os.listdir(settings.data_dir+'/train_x')
    num_test = int(len(all_path_x) * 0.1) 
    num_test = num_test - num_test%settings.ngpus + settings.ngpus
    all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
    ind = np.random.permutation(len(all_path_x))[0:num_test]
    for i in ind:
        os.rename('{}/train_x/{}'.format(settings.data_dir, all_path_x[i]), '{}/test_x/{}'.format(settings.data_dir, all_path_x[i]) )
        os.rename('{}/train_y/{}'.format(settings.data_dir, all_path_y[i]), '{}/test_y/{}'.format(settings.data_dir, all_path_y[i]) )

def get_noise_level(noise_level_tuple,noise_start_iter_tuple,iterations):
    assert len(noise_level_tuple) == len(noise_start_iter_tuple) and type(noise_level_tuple) in [tuple,list]
    noise_level = np.zeros(iterations+1)
    for i in range(len(noise_start_iter_tuple)-1):
        #remove this assert because it may not be necessary, and cause problem when iterations <3
        #assert i < iterations and noise_start_iter_tuple[i]< noise_start_iter_tuple[i+1]
        noise_level[noise_start_iter_tuple[i]:noise_start_iter_tuple[i+1]] = noise_level_tuple[i]
    assert noise_level_tuple[-1] < iterations 
    noise_level[noise_start_iter_tuple[-1]:] = noise_level_tuple[-1]
    return noise_level

def generate_first_iter_mrc(mrc,settings):
    '''
    Apply mw to the mrc and save as xx_iter00.xx
    '''
    root_name = mrc.split('/')[-1].split('.')[0]
    extension = mrc.split('/')[-1].split('.')[1]
    with mrcfile.open(mrc) as mrcData:
        orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)
    orig_data = apply_wedge(orig_data, ld1=1, ld2=0)
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)

    with mrcfile.new('{}/{}_iter00.{}'.format(settings.result_dir,root_name, extension), overwrite=True) as output_mrc:
        output_mrc.set_data(-orig_data)

    #preparation files for the first iteration
def prepare_first_iter(settings):
    if settings.preprocessing_ncpus >1:
        with Pool(settings.preprocessing_ncpus) as p:
            func = partial(generate_first_iter_mrc, settings=settings)
            p.map(func, settings.mrc_list)
    else:
        for i in settings.mrc_list:
            generate_first_iter_mrc(i,settings)
    return settings


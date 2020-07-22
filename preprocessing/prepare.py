import os 
import logging
import sys
import mrcfile
from mwr.preprocessing.cubes import create_cube_seeds,crop_cubes,DataCubes
from mwr.preprocessing.img_processing import normalize
from mwr.preprocessing.simulate import apply_wedge
from multiprocessing import Pool
import numpy as np
from functools import partial
from multiprocessing import Pool
from mwr.util.rotations import rotation_list

#Make a new folder. If exist, nenew it
def mkfolder(folder):
    import os
    try:
        os.makedirs(folder)
    except FileExistsError:
        print("Waring, the {} folder already exists before the 1st iteration".format(folder))
        print("The old {} folder will be removed".format(folder))
        import shutil
        shutil.rmtree(folder)
        os.makedirs(folder)

def generate_first_iter_mrc(mrc,settings):
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
    mkfolder(settings.result_dir)
    #if the input are tomograms
    if not settings.datas_are_subtomos:
        mkfolder(settings.subtomo_dir)
        #load the mask
        if settings.mask is not None:
            with mrcfile.open(settings.mask) as m:
                mask=m.data
        else:
            mask=None
        #use all the mrc files in the input folder
        if settings.input_dir[-1] == '\\' or settings.input_dir[-1] == '/':
            settings.input_dir= settings.input_dir[:-1]

        settings.tomogram_list = ["{}/{}".format(settings.input_dir,f) for f in os.listdir(settings.input_dir) if f.split(".")[-1]=="mrc" or f.split(".")[-1]=="rec" ]
        settings.tomogram_list_items = [f for f in os.listdir(settings.input_dir) if f.split(".")[-1]=="mrc" or f.split(".")[-1]=="rec"]
        settings.tomogram2_list = ["{}/{}".format(settings.input_dir,f) for f in os.listdir(settings.input_dir) if f.split(".")[-1]=="mrc2"]
        if len(settings.tomogram_list) <= 0:
            sys.exit("No input exists. Please check it in input folder!")
        if settings.tomogram2_list == []:
            settings.tomogram2_list = None
        for tomo_count, tomogram in enumerate(settings.tomogram_list):
            root_name = settings.tomogram_list_items[tomo_count].split('.')[0]
            with mrcfile.open(tomogram) as mrcData:
                orig_data = mrcData.data.astype(np.float32)
            seeds=create_cube_seeds(orig_data,settings.ncube,settings.cropsize,mask=mask)
            subtomos=crop_cubes(orig_data,seeds,settings.cropsize)

            for j,s in enumerate(subtomos):
                with mrcfile.new('{}/{}_{:0>6d}.mrc'.format(settings.subtomo_dir, root_name,j), overwrite=True) as output_mrc:
                    output_mrc.set_data(s.astype(np.float32))

            if settings.tomogram2_list is not None:
                #If the training is on the even odd tomograms, the second tomogram will be saved as subtomo_dir/*.mrc2
                with mrcfile.open(settings.tomogram2_list[tomo_count]) as mrcData:
                    orig_data = mrcData.data.astype(np.float32)

                subtomos=crop_cubes(orig_data,seeds,settings.cropsize)
                for j,s in enumerate(subtomos):
                    with mrcfile.new('{}/{}_{:0>6d}.mrc2'.format(settings.subtomo_dir, root_name,j), overwrite=True) as output_mrc:
                        output_mrc.set_data(s.astype(np.float32))

    settings.mrc_list = os.listdir(settings.subtomo_dir)
    settings.mrc_list = ['{}/{}'.format(settings.subtomo_dir,i) for i in settings.mrc_list]
    
    #need further test
    #with Pool(settings.preprocessing_ncpus) as p:
    #    func = partial(generate_first_iter_mrc, settings)
    #    res = p.map(func, settings.mrc_list)
    if settings.preprocessing_ncpus >1:
        with Pool(settings.preprocessing_ncpus) as p:
            func = partial(generate_first_iter_mrc, settings)
            res = p.map(func, settings.mrc_list)
    else:
        for i in settings.mrc_list:
            generate_first_iter_mrc(i,settings)


    if settings.tomogram2_list is not None:
        settings.mrc2_list=[]
        tmp=[]
        for m in settings.mrc_list:
            if m.endswith(".mrc2"):
                settings.mrc2_list.append(m)
            else:
                tmp.append(m)
        settings.mrc_list=tmp
        #should print mrc_list and mrc2_list
    else:
        settings.mrc2_list = None

    return settings
def get_cubes_one(data,settings, data2 = None, start = 0, mask = None, add_noise = 0):
    
    noise_factor = ((settings.iter_count - settings.noise_start_iter) // settings.noise_pause) +1 if settings.iter_count > settings.noise_start_iter else 0
 
    permutation = True
    if permutation and (data2 is not None):
        #randomly switch data1 and data2
        c = np.random.rand()

        if c > 0.5:
            if settings.combined_prediction:
                data2=(data+data2)/2.
            data_cubes = DataCubes(data, tomogram2 = data2, nCubesPerImg=1, cubeSideLen = settings.cube_sidelen, cropsize = settings.cropsize, mask = mask, noise_folder = settings.noise_folder,
            noise_level = settings.noise_level*noise_factor)
        else:
            if settings.combined_prediction:
                data=(data+data2)/2.
            data_cubes = DataCubes(data2, tomogram2 = data, nCubesPerImg=1, cubeSideLen = settings.cube_sidelen, cropsize = settings.cropsize, mask = mask, noise_folder = settings.noise_folder,
            noise_level = settings.noise_level*noise_factor)
    else:
        data_cubes = DataCubes(data, tomogram2 = data2, nCubesPerImg=1, cubeSideLen = settings.cube_sidelen, cropsize = settings.cropsize, mask = mask, noise_folder = settings.noise_folder,
    noise_level = settings.noise_level*noise_factor)
    for i,img in enumerate(data_cubes.cubesX):
        with mrcfile.new('{}/train_x/x_{}.mrc'.format(settings.data_folder, i+start), overwrite=True) as output_mrc:
            output_mrc.set_data(img.astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(settings.data_folder, i+start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_cubes.cubesY[i].astype(np.float32))
    return 0


def get_cubes(settings,inp):
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]
    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count)

    with mrcfile.open(current_mrc) as mrcData:
        ow_data = mrcData.data.astype(np.float32)*-1
    ow_data = normalize(ow_data, percentile = settings.normalize_percentile)
    with mrcfile.open('{}/{}_iter00.mrc'.format(settings.result_dir,root_name)) as mrcData:
        iw_data = mrcData.data.astype(np.float32)*-1
    iw_data = normalize(iw_data, percentile = settings.normalize_percentile)


    orig_data = apply_wedge(ow_data, ld1=0, ld2=1) + apply_wedge(iw_data, ld1 = 1, ld2=0)
    orig_data = normalize(orig_data, percentile = settings.normalize_percentile)


    if settings.tomogram2_list is not None :
        #TODO reduce redundancy
        current_mrc = '{}/{}_iter{:0>2d}.mrc2'.format(settings.result_dir,root_name, settings.iter_count)
        with mrcfile.open(current_mrc) as mrcData:
            ow_data = mrcData.data.astype(np.float32)*-1
        ow_data = normalize(ow_data, percentile = settings.normalize_percentile)
        with mrcfile.open('{}/{}_iter00.mrc2'.format(settings.result_dir,root_name)) as mrcData:
            iw_data = mrcData.data.astype(np.float32)*-1
        iw_data = normalize(iw_data, percentile = settings.normalize_percentile)
        orig_data2 = apply_wedge(ow_data, ld1=0, ld2=1) + apply_wedge(iw_data, ld1 = 1, ld2=0)
        orig_data2 = normalize(orig_data2, percentile = settings.normalize_percentile)

    for r in rotation_list:
        data = np.rot90(orig_data, k=r[0][1], axes=r[0][0])
        data = np.rot90(data, k=r[1][1], axes=r[1][0])
        if settings.tomogram2_list is not None:
            data2 = np.rot90(orig_data2, k=r[0][1], axes=r[0][0])
            data2 = np.rot90(data2, k=r[1][1], axes=r[1][0]) 
        else:
            data2=None
        get_cubes_one(data, settings, data2, start = start) 
        start += 1#settings.ncube

def get_cubes_list(settings):
    import os
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    if not os.path.exists(settings.data_folder):
        os.makedirs(settings.data_folder)
    for d in dirs_tomake:
        folder = '{}/{}'.format(settings.data_folder, d)
        if not os.path.exists(folder):
            os.makedirs(folder)
    inp=[]
    for i,mrc in enumerate(settings.mrc_list):
        inp.append((mrc, i*16))
    if settings.preprocessing_ncpus > 1:
        
        func = partial(get_cubes, settings)
        pool = Pool(processes=settings.preprocessing_ncpus) 
        logging.info('********{}'.format(len(inp)))
        res = pool.map(func,inp)
    if settings.preprocessing_ncpus == 1:
        logging.info('********{}'.format(len(inp)))
        for i in inp:
            get_cubes(settings,i)

    all_path_x = os.listdir(settings.data_folder+'/train_x')
    num_test = int(len(all_path_x) * 0.1)
    num_test = num_test - num_test%settings.ngpus + settings.ngpus
    all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
    ind = np.random.permutation(len(all_path_x))[0:num_test]
    for i in ind:
        os.rename('{}/train_x/{}'.format(settings.data_folder, all_path_x[i]), '{}/test_x/{}'.format(settings.data_folder, all_path_x[i]) )
        os.rename('{}/train_y/{}'.format(settings.data_folder, all_path_y[i]), '{}/test_y/{}'.format(settings.data_folder, all_path_y[i]) )
        #os.rename('data/train_y/'+all_path_y[i], 'data/test_y/'+all_path_y[i])

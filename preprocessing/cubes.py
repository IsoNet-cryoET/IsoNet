'''
Author: Hui Wang, Yuntao Liu
Date: 04/25/2020

Preprocessing functions for emapp 

--create_cube_seeds(): generate the seed array for creating cubes
--crop_cubes(): crop the cube from the entire tomograms
--prepare_cubes(): patch-wise generate train and test dataset


'''
import os
import glob
import numpy as np
import mrcfile

# from mwr.preprocessing.simulate import apply_wedge_dcube as apply_wedge
from mwr.preprocessing.simulate import apply_wedge

def create_cube_seeds(img3D,nCubesPerImg,cubeSideLen,mask=None):
    sp=img3D.shape
    if mask is None:
        cubeMask=np.ones(sp)
    else:
        cubeMask=mask
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((cubeSideLen,cubeSideLen,cubeSideLen), sp)])
    valid_inds = np.where(cubeMask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]), nCubesPerImg, replace=len(valid_inds[0]) < nCubesPerImg)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return (rand_inds[0],rand_inds[1], rand_inds[2])


def crop_cubes(img3D,seeds,cubeSideLen):
    size=len(seeds[0])
    cube_size=(cubeSideLen,cubeSideLen,cubeSideLen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,cube_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes


def prepare_cubes(X,Y,size=32,num=500):
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    #make folders for train and test dataset
    for d in dirs_tomake:
        try:
            os.makedirs('{}{}'.format(settings.ab_data_folder,d))
        except OSError:
            pass

    seeds=create_cube_seeds(X,num,size)

    subtomos_X=crop_cubes(X,seeds,size)
    subtomos_Y=crop_cubes(Y,seeds,size)

    for i,img in enumerate(subtomos_X):
        with mrcfile.new('{}train_x/x_{}.mrc'.format(settings.ab_data_folder, i), overwrite=True) as output_mrc:
            output_mrc.set_data(img.astype(np.float32))
        with mrcfile.new('{}train_y/y_{}.mrc'.format(settings.ab_data_folder, i), overwrite=True) as output_mrc:
            output_mrc.set_data(subtomos_Y[i].astype(np.float32))

    all_path_x = os.listdir('{}train_x/'.format(settings.ab_data_folder))
    num_test = int(len(all_path_x) * 0.1)
    if settings.ngpus > 1:
        num_test = num_test - num_test%settings.ngpus + settings.ngpus
    all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
    ind = np.random.permutation(len(all_path_x))[0:num_test]

    #seperate train and test dataset
    for i in ind:
        os.rename('{}train_x/{}'.format(settings.ab_data_folder, all_path_x[i]), '{}test_x/{}'.format(settings.ab_data_folder,all_path_x[i]))
        os.rename('{}train_y/{}'.format(settings.ab_data_folder, all_path_y[i]), '{}test_y/{}'.format(settings.ab_data_folder,all_path_y[i]))
        
    print("done create {} cubes! Split dataset into {} and {} for training and testing.".format(num,num-num_test,num_test))



class DataCubes:

    def __init__(self, tomogram, tomogram2 = None, nCubesPerImg=32, cubeSideLen=32, cropsize=32, mask = None, validationSplit=0.1, noise_folder = None, noise_level = 0.5):

        #TODO nCubesPerImg is always 1. We should not use this variable @Zhang Heng.
        #TODO consider add gaussian filter here
        self.tomogram = tomogram
        self.nCubesPerImg = nCubesPerImg
        self.cubeSideLen = cubeSideLen
        self.cropsize = cropsize
        self.mask = mask
        self.validationSplit = validationSplit
        self.__cubesY_padded = None
        self.__cubesX_padded = None
        self.__cubesY = None
        self.__cubesX = None
        self.noise_folder = noise_folder
        self.noise_level = noise_level

        #if we have two sub-tomograms for denoising (noise to noise), we will enable the parameter tomogram2, tomogram1 and 2 should be in same size
        #Using tomogram1 for X and tomogram2 for Y.
        self.tomogram2 = tomogram2
        self.__seeds = None

    @property
    def seeds(self):
        if self.__seeds is None:
            self.__seeds=create_cube_seeds(self.tomogram,self.nCubesPerImg,self.cropsize,self.mask)
        return self.__seeds

    @property
    def cubesX_padded(self):
        if self.__cubesX_padded is None:
            self.__cubesX_padded=crop_cubes(self.tomogram,self.seeds,self.cropsize).astype(np.float32)
            self.__cubesX_padded = np.array(list(map(apply_wedge, self.__cubesX_padded)), dtype = np.float32)
        return self.__cubesX_padded

    @property
    def cubesY_padded(self):
        if self.__cubesY_padded is None:
            if self.tomogram2 is None:
                self.__cubesY_padded=crop_cubes(self.tomogram,self.seeds,self.cropsize)
            else:
                self.__cubesY_padded=crop_cubes(self.tomogram2,self.seeds,self.cropsize)
            self.__cubesY_padded = self.__cubesY_padded.astype(np.float32)
        return self.__cubesY_padded


    @property
    def cubesY(self):
        if self.__cubesY is None:
            self.__cubesY = self.crop_to_size(self.cubesY_padded, self.cubeSideLen)
        return self.__cubesY

    @property
    def cubesX(self):
        if self.__cubesX is None:

            self.__cubesX = self.crop_to_size(self.cubesX_padded, self.cubeSideLen)
            if self.noise_folder is not None:
                if not os.path.exists(self.noise_folder):
                    os.makedirs(self.noise_folder)
                path_noise = sorted([self.noise_folder+'/'+f for f in os.listdir(self.noise_folder)])
                path_index = np.random.permutation(len(path_noise))[0:self.__cubesX.shape[0]]
                def read_vol(f):
                    with mrcfile.open(f) as mf:
                        res = mf.data
                    return res
                noise_volume = np.array([read_vol(path_noise[j]) for j in path_index])
                self.__cubesX += self.noise_level * noise_volume / np.std(noise_volume)

        return self.__cubesX


    def crop_to_size(self, array, size):
        start = self.cropsize//2 - size//2
        end = self.cropsize//2 + size//2
        return array[:,start:end,start:end,start:end]

    def create_training_data3D(self):
        n_val = int(self.cubesX.shape[0]*self.validationSplit)
        n_train = int(self.cubesX.shape[0])-n_val
        X_train, Y_train = self.cubesX[:n_train], self.cubesY[:n_train]
        X_train, Y_train = np.expand_dims(X_train,-1), np.expand_dims(Y_train,-1)
        X_test, Y_test = self.cubesX[-n_val:], self.cubesY[-n_val:]
        X_test, Y_test = np.expand_dims(X_test,-1), np.expand_dims(Y_test,-1)
        return (X_train, Y_train),(X_test, Y_test)


'''
save_training_data('data/my_training_data.npz', X, Y, XY_axes)

XY_axes="SCYX"

'''
import numpy as np
from tifffile import imsave,imread
from .image import *
from .filter import discard_slices
import mrcfile
class Tomogram:
    def __init__(self, volume):
        self.__volume = volume


    def rotate_aroundZ(self, angle_step = 60):
        '''
        Rotate self._dataX around Z
        '''
        from scipy.ndimage.interpolation import rotate
        rotX=self.__volume

        '''
        mp = True
        if mp:
            angles = np.arange(angle_step,360,angle_step)
            def rot(a):
                return rotate(rotX,a,(1,2),reshape=False)
            from multiprocessing import Pool
            p = Pool(10)
            result = p.map(rot,angles)
            print(len(result))
        '''

        for angle in range(angle_step,360,angle_step):
            print('rotate angle:',angle)
            rotated_X=rotate(rotX,angle,(1,2),reshape=False)
            rotX=np.concatenate((rotX,rotated_X))

        assert  len(self._dataX.shape)==3
        cropx=int(self._dataX.shape[2]*0.71//4*4)
        cropy=int(self._dataX.shape[1]*0.71//4*4)
        cropz=self._dataX.shape[0]

        #self.__volume=crop_center(self._dataX,cropx,cropy,cropz)

        return crop_center(rotX,cropx,cropy,cropz)

class DataPairs:

    #def __init__(self,fileNameX, fileNameY):
    #    '''read two 3D tomograms'''
    #    self._dataX=imread(fileNameX) # 4d
    #    self._dataY=imread(fileNameY)
    #    self.is_rotated=False

    def __init__(self):
        self.is_rotated=False


    def set_dataX(self,dataX):
        self._dataX=dataX

    def set_dataY(self,dataY):
        self._dataY=dataY


    def get_dataX(self):
        return self._dataX

    def get_dataY(self):
        return self._dataY

    def get_sliceNum(self):
        return self._dataX.shape[0]


    def create_patches(self,patchSideLen=128,nPatchesPerSlice=100,withFilter=None):
        #TODO patch_filter norm_percentile

        self._PatchesX = np.zeros([self._dataX.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        self._PatchesY = np.zeros([self._dataY.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        SliceNum = self._dataX.shape[0]
        print(self._dataX.shape)
        print(self._dataY.shape)
        for i in range(SliceNum):
            if withFilter==None:
                seedx,seedy = create_seed_2D(self._dataX[i],nPatchesPerSlice,patchSideLen)
            else: seedx,seedy = create_filter_seed_2D(self._dataX[i],nPatchesPerSlice,patchSideLen)
            self._PatchesX[i]=create_patch_image_2D(self._dataX[i],seedx,seedy,patchSideLen)
            self._PatchesY[i]=create_patch_image_2D(self._dataY[i],seedx,seedy,patchSideLen)
	#get patches by sliceNum*seedSize*y*x
        shp=self._PatchesX.shape
        self._PatchesX=self._PatchesX.reshape((shp[0]*shp[1],shp[2],shp[3]))
        self._PatchesY=self._PatchesY.reshape((shp[0]*shp[1],shp[2],shp[3]))

        #self._PatchesX=toUint8(self._PatchesX)
        #self._PatchesY=toUint8(self._PatchesY)
        return self._PatchesX,self._PatchesY

    def create_patches_new(self,patchSideLen=128,nPatchesPerSlice=100, mask = None):

        self._PatchesX = np.zeros([self._dataX.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        self._PatchesY = np.zeros([self._dataY.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        SliceNum = self._dataX.shape[0]
        print(self._dataX.shape)
        print(self._dataY.shape)
        for i in range(SliceNum):
            if mask==None:
                seedx,seedy = create_seed_2D(self._dataX[i],nPatchesPerSlice,patchSideLen)
            else: seedx,seedy = create_filter_seed_2D(self._dataX[i],nPatchesPerSlice,patchSideLen, mask = mask)
            self._PatchesX[i]=create_patch_image_2D(self._dataX[i],seedx,seedy,patchSideLen)
            self._PatchesY[i]=create_patch_image_2D(self._dataY[i],seedx,seedy,patchSideLen)
	#get patches by sliceNum*seedSize*y*x
        shp=self._PatchesX.shape
        self._PatchesX=self._PatchesX.reshape((shp[0]*shp[1],shp[2],shp[3]))
        self._PatchesY=self._PatchesY.reshape((shp[0]*shp[1],shp[2],shp[3]))

        #self._PatchesX=toUint8(self._PatchesX)
        #self._PatchesY=toUint8(self._PatchesY)
        return self._PatchesX,self._PatchesY

    def create_training_data2D(self, validationSplit=0.1):

        if not hasattr(self, '_PatchesY'):
            self._PatchesX = self._dataX
            self._PatchesY = self._dataY

        n_val=int(self._PatchesX.shape[0] * validationSplit)
        n_train= self._PatchesX.shape[0] - n_val
        X_t, Y_t = self._PatchesX[-n_val:],  self._PatchesY[-n_val:]
        shape1=X_t.shape
        X_t=X_t.reshape(*shape1,1)
        Y_t=Y_t.reshape(*shape1,1)
        X,   Y   = self._PatchesX[:n_train], self._PatchesY[:n_train]
        shape2=X.shape
        X=X.reshape(*shape2,1)
        Y=Y.reshape(*shape2,1)
        data_val = (X_t,Y_t)
        return (X,Y), data_val

    def create_training_data3D(self,validationSplit=0.1):
        n_val = int(self.cubesX.shape[0]*validationSplit)
        n_train = int(self.cubesX.shape[0])-n_val
        X_train, Y_train = self.cubesX[:n_train], self.cubesY[:n_train]
        X_train, Y_train = np.expand_dims(X_train,-1), np.expand_dims(Y_train,-1)
        X_test, Y_test = self.cubesX[-n_val:], self.cubesY[-n_val:]
        X_test, Y_test = np.expand_dims(X_test,-1), np.expand_dims(Y_test,-1)
        return (X_train, Y_train),(X_test, Y_test)


    def create_cubes(self,nCubesPerImg,cubeSideLen,mask=None):
        # self.cubesX=np.zeros([nCubesPerImg,cubeSideLen,cubeSideLen,cubeSideLen])
        # self.cubesY=np.zeros([nCubesPerImg,cubeSideLen,cubeSideLen,cubeSideLen])
        seeds=create_cube_seeds(self._dataX,nCubesPerImg,cubeSideLen,mask)
        self.cubesX=crop_cubes(self._dataX,seeds,cubeSideLen)
        self.cubesY=crop_cubes(self._dataY,seeds,cubeSideLen)
        return 0

    def rotate(self,step=60):
        from scipy.ndimage.interpolation import rotate
        rotX=self._dataX
        rotY=self._dataY
        for angle in range(step,360,step):
            print('rotate angle:',angle)
            rotated_X=rotate(rotX,angle,(1,2),reshape=False)
            rotated_Y=rotate(rotY,angle,(1,2),reshape=False)
            self._dataX=np.concatenate((self._dataX,rotated_X))
            self._dataY=np.concatenate((self._dataY,rotated_Y))
        assert  len(self._dataX.shape)==3
        cropx=int(self._dataX.shape[2]*0.71//4*4)
        cropy=int(self._dataX.shape[1]*0.71//4*4)
        cropz=self._dataX.shape[0]
        self._dataX=crop_center(self._dataX,cropx,cropy,cropz)
        self._dataY=crop_center(self._dataY,cropx,cropy,cropz)
        self.is_rotated=True
        return self._dataX,self._dataY



    def prefilter(self,start,end):
        self._dataX=discard_slices(self._dataX,start,end)
        self._dataY=discard_slices(self._dataY,start,end)
        return self._dataX,self._dataY



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
            from IsoNet.simulation.simulate import apply_wedge
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
                import os
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







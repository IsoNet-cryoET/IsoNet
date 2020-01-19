

'''
save_training_data('data/my_training_data.npz', X, Y, XY_axes)

XY_axes="SCYX"

'''
import numpy as np
from tifffile import imsave,imread

class DataPairs:

    #def __init__(self,fileNameX, fileNameY):
    #    '''read two 3D tomograms'''
    #    self._dataX=imread(fileNameX) # 4d
    #    self._dataY=imread(fileNameY)


    def get_dataX(self):
        return self._dataX

    def get_dataY(self):
        return self._dataY

    def set_dataX(self,dataX):
        self._dataX=dataX

    def set_dataY(self,dataY):
        self._dataY=dataY

    def get_sliceNum(self):
        return self._dataX.shape[0]
        

class DataPairs2D(DataPairs):

    def create_patches(self,patchSideLen=128,nPatchesPerSlice=100):
        #TODO patch_filter norm_percentile
        
        from mwr.util.image import create_patch_image_2D,create_seed_2D
        self._PatchesX = np.zeros([self._dataX.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        self._PatchesY = np.zeros([self._dataY.shape[0],nPatchesPerSlice,patchSideLen,patchSideLen])
        SliceNum = self._dataX.shape[0]
        print(self._dataX.shape)
        for i in range(SliceNum):
            seedx,seedy = create_seed_2D(self._dataX[i],nPatchesPerSlice,patchSideLen)

            self._PatchesX[i]=create_patch_image_2D(self._dataX[i],seedx,seedy,patchSideLen) 
            self._PatchesY[i]=create_patch_image_2D(self._dataY[i],seedx,seedy,patchSideLen) 
	#get patches by sliceNum*seedSize*y*x
        shp=self._PatchesX.shape
        self._PatchesX=self._PatchesX.reshape((shp[0]*shp[1],shp[2],shp[3],1))
        self._PatchesY=self._PatchesY.reshape((shp[0]*shp[1],shp[2],shp[3],1))
        return self._PatchesX,self._PatchesY
        
    def create_training_data(self, validationSplit=0.1):

        if not hasattr(self, '_PatchesY'):
            self.create_patches()

        n_val=int(self._PatchesX.shape[0] * validationSplit)
        n_train= self._PatchesX.shape[0] - n_val
        X_t, Y_t = self._PatchesX[-n_val:],  self._PatchesY[-n_val:]
        X,   Y   = self._PatchesX[:n_train], self._PatchesY[:n_train]
        data_val = (X_t,Y_t)
        return (X,Y), data_val


#    def save_training_data_new(self):
#        from csbdeep.io import save_training_data
#        save_training_data('data/my_training_data.npz', X, Y, XY_axes)

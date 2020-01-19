#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tifffile  import imread,imsave
from mwr.simulation.psf import TwoDPsf
'''  commented out at 2018.08.08 16:26 by zjj
a = TwoDPsf(50)
a.write()
from tifffile import imread
data=imread('p190-bin8-1.tif')
a.apply(data,'a.tif')
'''

from mwr.util.generate import DataPairs2D
from mwr.util import image



pairs = DataPairs2D('convoluted_of_a.tif','p190-bin8-1.tif')
SliceNum = pairs.get_sliceNum()
patchSideLen = 64                                                 #setting patch size
nPatchesPerSlice = 100                                           #setting patch number per slice
'''
patchesX = np.zeros([SliceNum,nPatchesPerSlice,patchSideLen,patchSideLen])
patchesY = np.zeros([SliceNum,nPatchesPerSlice,patchSideLen,patchSideLen])
'''

patchesX,patchesY = pairs.create_patches(patchSideLen,nPatchesPerSlice)
sh=patchesX.shape
patchesX=patchesX.reshape(sh[0],sh[1],sh[2])
patchesX=image.toUint8(patchesX)
print('patchesX.shape'+str(patchesX.shape))
imsave('patchesX.tif',patchesX)
#imsave('patchesY.tif',patchesY)

#(X,Y), data_val = pairs.create_training_data()
#print X.shape


#from mwr.model.net2D import train
#train(X,Y, data_val)
#from mwr.util import toUint8
#from tifffile import imsave

#imsave('patchesX_of_a'+'.tif',toUint8(patchesX))

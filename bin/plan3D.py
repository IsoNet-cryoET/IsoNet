'''3d psf 2d'''
# from mwr import get_orig_data, get_patches_3D
from scipy.ndimage.interpolation import rotate
import numpy as np
from mwr.simulation import TwoDPsf,TrDPsf
from mwr.util.generate import DataPairs
from tifffile import imread,imsave

def mrcRead(mrc):
    data=readMrcNumpy(mrc)
    header=readHeader(mrc)
    return data,header

def get_orig_data(fileName,fileType):
    if fileType=='rec':
        data,header=mrcRead(fileName)
    else:
        data=imread(fileName)
    return data

def get_patches_3D(orig_data,outName='train_and_test_data.npz',npatchesper=100,patches_sidelen=128,torotate=False,prefilter=None \
    ,noisefilter=False,type=None):
    # imgpre.imsave(filename+'_convoluted',convoluted)
    pair=DataPairs()
    pair.set_dataY(orig_data)
    pair.set_dataX(orig_data)
    if prefilter!=None:
        pair.prefilter(prefilter[0],prefilter[1])
    if torotate==True :
        pair.rotate()
    threeD_missingwedge = TrDPsf(128)

    rotate_before = rotate(orig_data,90,axes=(0,2))

    conved = threeD_missingwedge.apply(np.expand_dims(rotate_before, axis=0))
    conved = np.squeeze(conved, axis=0)

    rotate_after = rotate(conved,-90,axes=(0,2))
    imsave('conved3D.tif',rotate_after.astype(np.uint8))
    pair.set_dataX(rotate_after)
    pair.create_patches(patches_sidelen,npatchesper,withFilter=filter)
    train_data,test_data=pair.create_training_data2D()
    print ('train_data.shape:',train_data[0].shape)
    np.savez(outName,train_data=train_data,test_data=test_data)


filename = '/home/heng/mwr/test_data/test3D/GT/corrected_last1.tif'
savename = 'conved3dpatches.npz'
type = 'tif'

orig_data = get_orig_data(filename,type)
get_patches_3D(orig_data,outName=savename,npatchesper=100,patches_sidelen=128,torotate=False,prefilter=None \
    ,noisefilter=False,type=None)

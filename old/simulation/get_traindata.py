'''
run to get training data, which is stored as npz, named ???
 parameter:
 fileName, nstorepath,
'''
import imgpre
import numpy as np
import sys


def get_paches(filename,storepath=None,npathcesper=100,patches_sidelen=128,rotate=True,prefilter=None \
    ,noisefilter=False,type=None):
    orig_data=imgpre.imread(filename)
    twoD_missingwedge=z.simulate.TwoDPsf(orig_data.shape[1],orig_data.shape[2])
    convoluted=twoD_missingwedge.apply(orig_data)
    imgpre.imsave(filename+'_convoluted',convoluted)
    pair=imgpre.DataPairs(filename,filename+'_convoluted')
    if prefilter!=None:
        pair.prefilter(prefilter[0],prefilter[1])
    if rotate==True :
        pair.rotate()
    pair.create_patches(patches_sidelen,npathcesper,withFilter=filter)
    train_data,test_data=pair.create_training_data2D()
    print ('train_data.shape:',train_data[0].shape)
    np.savez(storepath+'/'+'train_and_test_data.npz',train_data=train_data,test_data=test_data)

if __name__=='__main__':
    get_paches('/home/heng/p190-bin8-1',rotate=False,prefilter=(30,70))

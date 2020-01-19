#!/usr/bin/env python3

from mwr.util import readMrcNumpy,readHeader,PercentileNormalizer
from mwr.simulation import TwoDPsf,TrDPsf
from mwr.util.generate import DataPairs
import numpy as np
from tifffile import imread,imsave
from mwr.models import unet
from scipy.ndimage.interpolation import rotate
import os
from setting import Setting


def mrcRead(mrc):
    data=readMrcNumpy(mrc)
    header=readHeader(mrc)
    return data,header

def rotate_before(real_data,direction):
    if direction == 'X' or direction =='x':
        return np.rot90(real_data, 1, (0,1))
    elif direction == 'Y' or direction =='y':
        return np.rot90(real_data, 1, (0,2))
    elif direction == 'Z' or direction == 'z':
        return np.rot90(real_data, 1, (1,2))
    else:
        return real_data
def rotate_after(out_data,direction):
    if direction == 'X' or direction =='x':
        return np.rot90(out_data, 1, (1,0))
    elif direction == 'Y' or direction =='y':
        return np.rot90(out_data, 1, (2,0))
    elif direction == 'Z' or direction == 'z':
        return np.rot90(out_data, 1, (2,1))
    else:
        return out_data

def get_patches(orig_data,outName='train_and_test_data.npz',npatchesper=100,patches_sidelen=128,rotate=False,prefilter=None \
    ,noisefilter=False,type=None):

    #imgpre.imsave(filename+'_convoluted',convoluted)

    pair=DataPairs()
    pair.set_dataY(orig_data)
    pair.set_dataX(orig_data)
    if prefilter!=None:
        pair.prefilter(prefilter[0],prefilter[1])
    if rotate==True :
        pair.rotate()
    sp=pair.get_dataX().shape
    twoD_missingwedge=TwoDPsf(sp[1],sp[2])
    pair.set_dataX(twoD_missingwedge.apply(pair.get_dataX()))
    pair.create_patches(patches_sidelen,npatchesper,withFilter=filter)
    train_data,test_data=pair.create_training_data2D()
    print ('train_data.shape:',train_data[0].shape)
    np.savez(outName,train_data=train_data,test_data=test_data)


def get_patches_3D(orig_data,outName='train_and_test_data.npz',npatchesper=100,patches_sidelen=128,rotate=False,prefilter=None \
    ,noisefilter=False,type=None,rot_axis=(0,2)):
    # imgpre.imsave(filename+'_convoluted',convoluted)
    pair=DataPairs()
    pair.set_dataY(orig_data)
    pair.set_dataX(orig_data)
    if prefilter!=None:
        pair.prefilter(prefilter[0],prefilter[1])
    if rotate==True :
        pair.rotate()
    threeD_missingwedge = TrDPsf(128)
    rotate_before = np.rot90(pair.get_dataX(),1,rot_axis)
    conved = threeD_missingwedge.apply(np.expand_dims(rotate_before, axis=0))
    conved = np.squeeze(conved, axis=0)

    rotate_after = np.rot90(conved,3,rot_axis)
    pair.set_dataX(rotate_after)
    #imsave('conved'+str(rot_axis),rotate_after)
    pair.create_patches(patches_sidelen,npatchesper,withFilter=filter)
    train_data,test_data=pair.create_training_data2D()
    print ('train_data.shape:',train_data[0].shape)
    np.savez(outName,train_data=train_data,test_data=test_data)


def get_cubes(orig_data,outName,ncube=1000,cube_sidelen=64):
    pair = DataPairs()
    pair.set_dataX(orig_data)
    pair.set_dataY(orig_data)
    pair.create_cubes(nCubesPerImg=ncube,cubeSideLen=cube_sidelen)
    threeD_missingwedge=TrDPsf(cube_sidelen)
    print('cubes.shape',pair.cubesX.shape)
    pair.cubesX=threeD_missingwedge.apply(pair.cubesX)
    train_data, test_data=pair.create_training_data3D()
    print ('train_data.shape:', train_data[0].shape)
    np.savez(outName, train_data=train_data, test_data=test_data)

def get_cubes_wholeconv(orig_data,outName,ncube,cube_sidelen):
    # threeD_missingwedge=TrDPsf(cube_sidelen)
    # print('convoluting')
    # conved = threeD_missingwedge.apply(np.expand_dims(orig_data,axis=0))
    # conved = np.squeeze(conved,axis=0)
    pair = DataPairs()
    pair.set_dataX(orig_data)
    pair.set_dataY(orig_data)
    pair.create_cubes(nCubesPerImg=ncube,cubeSideLen=cube_sidelen)
    train_data, test_data=pair.create_training_data3D()
    print ('train_data.shape:', train_data[0].shape)
    np.savez(outName, train_data=train_data, test_data=test_data)


def train_data(fileName, outFile,epochs=40,batch_size=32,steps_per_epoch=28,n_gpus=2):
    data = np.load(fileName)
    Normalizer = PercentileNormalizer()
    (x,y,x_val,y_val) = (data['train_data'][0],data['train_data'][1],data['test_data'][0],data['test_data'][1])
    x = Normalizer.before(x,'ZYXC')
    y = Normalizer.before(y,'ZYXC')
    x_val = Normalizer.before(x_val,'ZYXC')
    y_val = Normalizer.before(y_val,'ZYXC')
    unet.train(x,y,(x_val,y_val),outFile,epochs=epochs,steps_per_epoch=steps_per_epoch,batch_size=batch_size,n_gpus=n_gpus)


def train_data3D(fileName, outFile,epochs=40,batch_size=32,steps_per_epoch=28, n_gpus=2):
    data = np.load(fileName)
    (x,y,x_val,y_val) = (data['train_data'][0],data['train_data'][1],data['test_data'][0],data['test_data'][1])
    Normalizer = PercentileNormalizer()
    x = Normalizer.before(x,'SZYXC')
    y = Normalizer.before(y,'SZYXC')
    x_val = Normalizer.before(x_val,'SZYXC')
    y_val = Normalizer.before(y_val,'SZYXC')
    print('x shape:',x.shape)
    unet.train3D(x,y,(x_val,y_val),outFile,epochs=epochs,steps_per_epoch=steps_per_epoch,batch_size=batch_size,n_gpus=n_gpus)

def get_orig_data(fileName,fileType):
    if fileType=='rec':
        data,header=mrcRead(fileName)
    else:
        data=imread(fileName)
    return data


def prepare_and_train2D(fileName, fileType, npzfile,weightName,n_gpus,settings):
    if fileName is None:
        train_data(npzfile+'_YZ.npz', weightName,epochs=settings.epoch,batch_size=settings.batch_size,
                   steps_per_epoch=settings.steps_per_epoch, n_gpus=n_gpus)
    else:
        data = get_orig_data(fileName, fileType)
        num_image = data.shape[0]
        data = data[int(0.1 * num_image):int(0.9 * num_image)]
        get_patches(data, outName=npzfile, npatchesper=settings.npatchesper,
                    patches_sidelen=settings.patches_sidelen, rotate=settings.rotate, prefilter=None , noisefilter=False, type=None)
        train_data(npzfile, weightName, epochs=settings.epoch, batch_size=settings.batch_size,
                   steps_per_epoch=settings.steps_per_epoch, n_gpus=n_gpus)


def prepare_and_train3D(fileName, fileType, npzfile,weightName,n_gpus,settings,direction):
    if fileName is None:
        train_data3D(npzfile, weightName,epochs=settings.epoch, n_gpus=n_gpus,
                     batch_size=settings.batch_size, steps_per_epoch=settings.steps_per_epoch)
    else:
        orig_data = get_orig_data(fileName,fileType)
        # to get xyz image
        data_rotated = rotate_before(orig_data,direction)
        num_image = data_rotated.shape[2]
        data_rotated = data_rotated[:, :, int(0.05 * num_image):int(0.95 * num_image)]
        print('data_rotated.shape:', data_rotated.shape)
        # imsave('GT_xzy.tif', data_rotated)
        get_cubes_wholeconv(data_rotated, outName=npzfile, ncube=settings.ncube, cube_sidelen=settings.cube_sidelen)
        print('cubes got')
        train_data3D(npzfile, weightName, epochs=settings.epoch, n_gpus=n_gpus,
                     batch_size=settings.batch_size, steps_per_epoch=settings.steps_per_epoch)

def average_train(fileName, fileType, npzfile,weightName,n_gpus,settings):
    orig_data = get_orig_data(fileName, fileType)
    #get_patches_3D(orig_data, outName=str(npzfile)+'_YZ.npz', npatchesper=settings.npatchesper, rotate=settings.rotate, rot_axis=(0, 2))
    #get_patches_3D(orig_data, outName=str(npzfile)+'_XZ.npz', npatchesper=settings.npatchesper, rotate=settings.rotate, rot_axis=(0, 1))
    train_data(npzfile+'_YZ.npz', weightName+'_YZ.npz',epochs=settings.epoch,batch_size=settings.batch_size,
                   steps_per_epoch=settings.steps_per_epoch, n_gpus=n_gpus)
    train_data(npzfile+'_XZ.npz', weightName+'_XZ.npz',epochs=settings.epoch,batch_size=settings.batch_size,
                   steps_per_epoch=settings.steps_per_epoch, n_gpus=n_gpus)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, default=None, help='Your mrc file')
    parser.add_argument('--weight', type=str, default='weights_last.h5' ,help='Weight file name to save')
    parser.add_argument('--data', type=str, default='train_and_test_data' ,help='Data file name to save')
    parser.add_argument('--type', type=str, default='rec', help='type of data .mrc or .tif')
    parser.add_argument('--gpus', type=int, default=2, help='number of gpu fro training')
    parser.add_argument('--dim', type=str, default='2D', help='training 2D or 3D')
    parser.add_argument('--ip', type=str, default=None, help='to specify working gpus')
    parser.add_argument('--direc', type=str, default=None, help='rotate axis when 3D simulating')
    args = parser.parse_args()
    settings = Setting()
    if args.ip == '24':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1,2"  # specify which GPU(s) to be used

    if args.dim=='2D':
        average_train(args.file,args.type,args.data,args.weight,args.gpus,settings)
        #prepare_and_train2D(args.file,args.type,args.data,args.weight,args.gpus,settings)
    elif args.dim == '3D':
        prepare_and_train3D(args.file,args.type,args.data,args.weight,args.gpus,settings,args.direc)

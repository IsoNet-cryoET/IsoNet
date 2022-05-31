
#!/usr/bin/env python
# encoding: utf-8

import numpy as np

def mw2d(dim,missingAngle=[30,30]):
    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2
            #elif x==0 and y==0:
            #    theta=0
            #elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=min(dim/2,dim/2)**2:
                if x > 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1
    #from mwr.util.image import norm_save
    #norm_save('mw.tif',self._mw)
    return mw


#import tensorflow as tf
def apply_wedge_dcube(ori_data, mw2d, mw3d=None):
    if mw3d is None:
        #if len(ori_data.shape) > 3:
        #    ori_data = np.squeeze(ori_data, axis=-1)
        data = np.rot90(ori_data, k=1, axes=(1,2)) #clock wise of counter clockwise??
        data = np.fft.ifft2(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * np.fft.fft2(data))
        #data = tf.signal.ifft2d(np.fft.fftshift(mw2d)[np.newaxis, np.newaxis, :, :] * tf.signal.fft2d(data))
        data = np.real(data)
        data=np.rot90(data, k=3, axes=(1,2))

    else:
        import mrcfile
        with mrcfile.open(mw3d, 'r') as mrc:
            mw = mrc.data
        mwshift = np.fft.fftshift(mw)
        data = np.zeros_like(ori_data)
        for i,d in enumerate(ori_data):
            f_data = np.fft.fftn(d)
            outData = mwshift*f_data
            inv = np.fft.ifftn(outData)
            data[i] = np.real(inv).astype(np.float32)
            #data[i] = normalize(data[i],percentile=True)

    return data

def apply_wedge(ori_data, ld1 = 1, ld2 =0):
    #apply -60~+60 wedge to single cube
    data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
    mw = TwoDPsf(data.shape[1], data.shape[2]).getMW()

    #if inverse:
    #    mw = 1-mw
    mw = mw * ld1 + (1-mw) * ld2

    mw3d = np.zeros(data.shape,dtype=np.complex)
    f_data = np.fft.fftn(data)
    for i, item in enumerate(f_data):
        mw3d[i] = mw
    mwshift = np.fft.fftshift(mw)
    outData = mwshift*f_data
    inv = np.fft.ifftn(outData)
    real = np.real(inv).astype(np.float32)
    out = np.rot90(real, k=3, axes=(0,1))
    return out

def apply_wedge1(ori_data, ld1 = 1, ld2 =0, mw3d = None):

    if mw3d is None:
        data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
        mw = mw2d(data.shape[1])

        #if inverse:
        #    mw = 1-mw
        mw = mw * ld1 + (1-mw) * ld2

        outData = np.zeros(data.shape,dtype=np.float32)
        mw_shifted = np.fft.fftshift(mw)
        for i, item in enumerate(data):
            outData_i=np.fft.ifft2(mw_shifted * np.fft.fft2(item))
            outData[i] = np.real(outData_i)

        outData.astype(np.float32)
        outData=np.rot90(outData, k=3, axes=(0,1))
        return outData
    else:
        import mrcfile
        with mrcfile.open(mw3d, 'r') as mrc:
            mw = mrc.data
        mw = np.fft.fftshift(mw)
        mw = mw * ld1 + (1-mw) * ld2

        f_data = np.fft.fftn(ori_data)
        outData = mw*f_data
        inv = np.fft.ifftn(outData)
        outData = np.real(inv).astype(np.float32)
    return outData

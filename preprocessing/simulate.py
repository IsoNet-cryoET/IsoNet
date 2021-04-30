
#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from scipy.signal import convolve
import datetime
#from preprocessing.img_processing import *

class TwoDPsf:
    def __init__(self,size_y,size_x):
        #TODO Normalize
        self._dimension=(size_y,size_x)
        self._mw=self.getMW()
        # shiftedMW = np.fft.fftshift(self._mw)
        # self._psf=np.fft.fft2(shiftedMW)
        # self._psf=np.fft.fftshift(self._psf)
        # sum = np.sum(self._psf)
        # self._psf=self._psf/np.sum(self._psf)  # Normalize done at 20180830 by zjj

    def getMW(self,missingAngle=[30,30]):
        self._mw=np.zeros((self._dimension[0],self._dimension[1]),dtype=np.double)
        missingAngle = np.array(missingAngle)
        missing=np.pi/180*(90-missingAngle)
        for i in range(self._dimension[0]):
            for j in range(self._dimension[1]):
                y=(i-self._dimension[0]/2)
                x=(j-self._dimension[1]/2)
                if x==0:# and y!=0:
                    theta=np.pi/2
                #elif x==0 and y==0:
                #    theta=0
                #elif x!=0 and y==0:
                #    theta=np.pi/2
                else:
                    theta=abs(np.arctan(y/x))

                if x**2+y**2<=min(self._dimension[0]/2,self._dimension[1]/2)**2:
                    if x > 0 and y > 0 and theta < missing[0]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x < 0 and y < 0 and theta < missing[0]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x > 0 and y < 0 and theta < missing[1]:
                        self._mw[i,j]=1#np.cos(theta)
                    if x < 0 and y > 0 and theta < missing[1]:
                        self._mw[i,j]=1#np.cos(theta)

                if int(y) == 0:
                    self._mw[i,j]=1
        #from mwr.util.image import norm_save
        #norm_save('mw.tif',self._mw)
        return self._mw

    def circleMask(self):
        dim0 = self._dimension[0]
        dim1 = self._dimension[1]
        s0 = np.arange(-int(dim0/2),dim0-int(dim0/2))
        s1 = np.arange(-int(dim1/2),dim1-int(dim1/2))
        y,x = np.meshgrid(s1,s0)
        circle = np.array(4*x**2/(dim0**2) + 4*y**2/(dim1**2) < 1)
        return circle.astype(np.uint8)

    #
    # def apply(self,data,name):
    #     from image import crop_center,toUint8
    #
    #     # data=np.pad(data,pad_width=((0,0),(self._dimension,self._dimension),(self._dimension,self._dimension)),mode='median')
    #
    #     outData=np.zeros(data.shape)
    #     y,x = outData.shape[1],outData.shape[2]
    #     for i,item in enumerate(data):
    #         res=fftconvolve(item,self._psf,mode='same')
    #         res=np.real(res)
    #         outData[i]=res
    #         #print(i)
    #     outData=toUint8(outData)
    #     imsave('convoluted'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'of'+name,crop_center(outData,x-2*self._dimension,y-2*self._dimension))
    
    def apply(self,data):
    #inp: 3D;a stack of 2D images
        from ..util.fft import fft2_gpu, ifft2_gpu
        outData = np.zeros(data.shape,dtype=np.float32)
        for i, item in enumerate(data):

            assert self._mw.shape==item.shape
            # mw=np.fft.ifft2(np.fft.fftshift(self._mw))
            # mw=np.fft.fftshift(mw)
            # outData_i=convolve(data[i],mw,mode='same')

            outData_i=np.fft.fftshift(self._mw) * fft2_gpu(item, fftshift=False)
            outData_i=ifft2_gpu(outData_i, fftshift=False)
            #outData_i = np.fft.ifft2(fou)
            outData[i] = np.real(outData_i)#.astype(np.uint8)
        return outData

    def apply_old(self,data):

        outData = np.zeros(data.shape,dtype=np.float32)
        for i, item in enumerate(data):
            #print(i)
            assert self._mw.shape==item.shape
            # mw=np.fft.ifft2(np.fft.fftshift(self._mw))
            # mw=np.fft.fftshift(mw)
            # outData_i=convolve(data[i],mw,mode='same')
            outData_i=np.fft.ifft2(np.fft.fftshift(self._mw) * np.fft.fft2(item))
            #outData_i = np.fft.ifft2(fou)
            outData[i] = np.real(outData_i)#.astype(np.uint8)
        return outData


class TrDPsf:
    def __init__(self,sideLen):
        self.sideLen=sideLen
        self.mw=self.getMw3D()

    def getMw3D(self,missingAngle=30):
        self.mw=np.zeros([self.sideLen,self.sideLen,self.sideLen],dtype=np.float32)
        theta=np.pi/180*(90-missingAngle)
        for i in range(self.sideLen):
            z = (i - self.sideLen // 2)
            for j in range(self.sideLen):
                y = (j - self.sideLen // 2)
                for k in range(self.sideLen):
                    x=(k-self.sideLen//2)
                    if abs(y)<=abs(x*np.tan(theta)) and x*x+y*y+z*z<(self.sideLen//2)*(self.sideLen//2):
                        self.mw[j,i,k]=1
                    else:
                        self.mw[j,i,k]=0
        return self.mw
'''
    def apply(self,data):

        data = np.expand_dims( data, axis=0)

        mask=np.fft.fftshift(self.mw)
        mask=np.fft.ifftn(mask)
        mask=np.fft.fftshift(mask)
        sp=data.shape
        data = np.pad(data,pad_width=((0,0),(self.sideLen,self.sideLen),(self.sideLen,self.sideLen),(self.sideLen,self.sideLen)),mode='edge')
        outData=np.zeros(sp, dtype=np.float32)
        assert len(data.shape)==4
        for i ,item in enumerate(data):
            #print('1.0')
            res=convolve(item,mask,mode='same')
            res=np.real(res)
            #print('1.1')
            outData[i]=res[self.sideLen+1:sp[1]+self.sideLen+1,self.sideLen+1:sp[2]+self.sideLen+1,self.sideLen+1:sp[3]+self.sideLen+1]
            #print('1.2')
            #outData = outData.astype(np.uint8)

        return np.squeeze(outData, axis=0)




def apply_wedge1(ori_data, ld1 = 1, ld2 =0):

    data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
    mw = TwoDPsf(data.shape[1], data.shape[2]).getMW()

    #if inverse:
    #    mw = 1-mw
    mw = mw * ld1 + (1-mw) * ld2

    outData = np.zeros(data.shape,dtype=np.float32)
    for i, item in enumerate(data):
        outData_i=np.fft.ifft2(np.fft.fftshift(mw) * np.fft.fft2(item))
        outData[i] = np.real(outData_i)

    outData.astype(np.float32)
    outData=np.rot90(outData, k=3, axes=(0,1))
    return outData
'''
def apply_wedge_dcube(ori_data):
    if len(ori_data.shape) > 3:
        ori_data = np.squeeze(ori_data, axis=-1)
    #print("2.2.1")
    data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
    #print("2.2.2")
    t = TwoDPsf(data.shape[1], data.shape[2])
    data = t.apply_old(data)
    #print("2.2.3")
    data=np.rot90(data, k=3, axes=(0,1))
    #print("2.2.4")
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

def apply_wedge1(ori_data, ld1 = 1, ld2 =0):

    data = np.rot90(ori_data, k=1, axes=(0,1)) #clock wise of counter clockwise??
    mw = TwoDPsf(data.shape[1], data.shape[2]).getMW()

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

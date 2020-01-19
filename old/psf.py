
#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tifffile import imsave,imread
from scipy.signal import fftconvolve
import datetime


	

class Psf:
    def __init__(self):
        pass

    def write(self):
        from mwr.util import toUint8
        out1=toUint8(self._psf)
        imsave("fft.tif",out1)

    def getDimension(self):
        return self._dimension


    

class TwoDPsf(Psf):
    def __init__(self,dimension):
        #TODO Normalize  
        self._dimension=dimension
        self.getMW()
        shiftedMW = np.fft.fftshift(self._mw)
        self._psf=np.fft.fft2(shiftedMW)
        self._psf=np.fft.fftshift(self._psf)
        sum = np.sum(self._psf)
        self._psf=self._psf/np.sum(self._psf)  # Normalize done at 20180830 by zjj
        
    def getMW(self):
        self._mw=np.zeros((self._dimension,self._dimension),dtype=np.double)

        for i in range(self._dimension):
            for j in range(self._dimension):
                x=(i-self._dimension/2)
                y=(j-self._dimension/2)
                if (y-x*np.tan(np.pi/180*(90-60)))*\
                    (y+x*np.tan(np.pi/180*(90-60)))>=0 and x**2+y**2<(self._dimension/2)**2:
                    if y*x>0:
                        theta=np.pi/2-np.arctan(y/float(x))
                    elif y*x<0:
                        theta=np.pi/2+np.arctan(y/float(x))
                    elif x==0:
                        theta=0

                    weight=np.sqrt(np.cos(theta))
                    if (i-self._dimension/2.0)**2+(j-self._dimension/2.0)**2<(self._dimension/2.0)**2:
                        self._mw[i,j]=weight
        return self._mw


    def apply(self,data,name):
        from mwr.util import crop_center,toUint8
        outData=np.zeros(data.shape)  
        y,x = outData.shape[1],outData.shape[2]
        for i,item in enumerate(data):
            res=fftconvolve(item,self._psf,mode='same')
            res=np.real(res)
            outData[i]=res
            print('changed')
        outData=toUint8(outData)
        imsave('convoluted'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'of'+name,outData)

    

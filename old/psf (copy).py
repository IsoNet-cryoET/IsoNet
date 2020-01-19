#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from tifffile import imsave,imread

def toUint8(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*200
    data=data.astype(np.uint8)
    return data

def crop_center(img,cropx,cropy):
    y,x = img.shape[1],img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]

dimension=50
twoDctf=np.zeros((dimension,dimension),dtype=np.double)

for i in xrange(dimension):
    for j in xrange(dimension):
        x=(i-dimension/2)
        y=(j-dimension/2)
        if (y-x*np.tan(np.pi/180*(90-60)))*\
            (y+x*np.tan(np.pi/180*(90-60)))>=0 and x**2+y**2<(dimension/2)**2:
            if y*x>0:
                theta=np.pi/2-np.arctan(y/float(x))
            elif y*x<0:
                theta=np.pi/2+np.arctan(y/float(x))
            elif x==0:
                theta=0
            print theta*180/np.pi
            weight=np.sqrt(np.cos(theta))
            if (i-dimension/2.0)**2+(j-dimension/2.0)**2<(dimension/2.0)**2:
                twoDctf[i,j]=weight
            #if x==0:
            #    twoDctf[i,j]=1

out1=toUint8(twoDctf)
imsave('a.tif',out1)
#import scipy.ndimage as ndimage
#twoDctf = ndimage.gaussian_filter(twoDctf, sigma=(0.5,0.5), order=0)
twoDctf = np.fft.fftshift(twoDctf)




twoDpsf=np.fft.fft2(twoDctf)
twoDpsf=np.fft.fftshift(twoDpsf)



#print twoDpsf
fft=toUint8(twoDpsf)
imsave("fft.tif",fft)



from scipy.ndimage.interpolation import rotate

def getSource(oriImg,outImg):
    realData=imread(oriImg)
    realData=realData.astype(np.double)
    realData=toUint8(realData)
    shapeY=realData.shape[1]
    shapeX=realData.shape[2]
    cropX=int(shapeX*0.75)
    cropY=int(shapeY*0.75)
    #imsave('p190-bin8-1.tif',realData)

    realData=np.pad(realData,pad_width=((0,0),(50,50),(50,50)),mode='median')
    realData1=np.zeros(realData.shape,dtype=np.float)
    realData2=np.copy(realData)
    for angle in range(10,360,10):
        print angle
        realData1=rotate(realData, angle, axes=(1,2),reshape=False)
        realData2=np.concatenate((realData2,realData1))
    imsave('../data/origin/'+outImg,crop_center(realData2,cropX,cropY))

    from scipy.signal import fftconvolve
    outData=np.zeros(realData2.shape)
    for i,item in enumerate(realData2):
        res=fftconvolve(item,twoDpsf,mode='same')
        res=np.real(res)
        outData[i]=res
        print i
    outData=toUint8(outData)
    imsave('../data/wedged/'+outImg,crop_center(outData,cropX,cropY))

import os 

path = '../../dataTif'  
for i in os.listdir(path):  
    print i
    getSource(path+"/"+i,'source_'+i) 


#!/usr/bin/env python3
'''
perform a lowpass filter before mwr
threshold is percentage of lowpass diameter to the sidelen of image
'''
import numpy as np 
import mrcfile
# from mwr.util import fft

def circle_mask(h,w,p):
    mask=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if (i-h/2)**2/h**2+(j-w/2)**2/w**2<p**2/4:
                mask[i,j]=1
    return mask

def sphere_mask(l,w,h,p):
    mask=np.zeros((l,w,h))
    #rad=int(np.min((int(h*p),int(w*p),int(l*p))))
    for i in range(l):
        for j in range(w):
            for k in range(h):

                if (i-l/2)**2/l**2+(j-w/2)**2/w**2+(k-h/2)**2/h**2<p**2/4:
                    mask[i,j,k]=1
    return mask
if __name__ == "__main__":
    import sys
    args=sys.argv
    data=mrcfile.open(args[1]).data
    sp=data.shape
    mask=sphere_mask(sp[0],sp[1],sp[2],float(args[3]))
    mask=mask.astype(np.float32)
    save=np.zeros(sp)
    data_f=np.fft.fftn(data)
    data_fsf=np.fft.fftshift(data_f)
    masked=mask*data_fsf
    masked=np.fft.fftshift(masked)
    save=np.real(np.fft.ifftn(masked))
    
    # for z in range(sp[0]):
    #     img=data[z]
    #     ft=fft.fft2_gpu(img,fftshift=True)
    #     ft=ft*mask
    #     img=fft.ifft2_gpu(ft,fftshift=True)
    #     save[z]=np.real(img).astype(np.float32)
    #     print(z)
    save=save.astype(np.float32)
    with mrcfile.new(args[2],overwrite=True) as n:
        n.set_data(save)
    with mrcfile.new('mask.rec',overwrite=True) as f:
        f.set_data(mask)



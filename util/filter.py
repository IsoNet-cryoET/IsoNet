"""
Generate mask by comparing local variance and global variance
"""
import numpy as np

def maxmask(tomo, side=5,percentile=60):
    from scipy.ndimage.filters import maximum_filter, median_filter, gaussian_filter
    print('Gaussian_filter')
    tomo = tomo.astype(np.float32)
    image1 = gaussian_filter(tomo, side/2)
    # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
    # patch_size = [(p//2 if p>1 else p) for p in patch_size]
    print('maximum_filter')
    filtered = maximum_filter(-image1, 2*side+1, mode='reflect')
    out =  filtered > np.percentile(-image1,100-percentile)
    out = out.astype(np.uint8)
    return out


def stdmask_old(tomo,cubelen=20,std=None):
    #use std of tomo as threshold if surrunding pixels'std > tomo_std, mark 1;else 0
    if std is not None:
        tomo_std = std
    else:
        tomo_std = np.std(tomo)
    sp = tomo.shape
    dim=len(sp)
    mask = np.zeros(sp)
    half_len=int(cubelen/2)
    if dim==3:
        padded = np.pad(tomo,((half_len,half_len),(half_len,half_len),(half_len,half_len)),'reflect')
    elif dim==4:
        padded = np.pad(tomo,((half_len,half_len),(half_len,half_len),(half_len,half_len),(0,0)),'reflect')
    else:
        raise Exception("stdmask: dim is wrong")

    for i in range(sp[0]):
        for j in range(sp[1]):
            for k in range(sp[2]):
                if np.std(padded[(i):(i+2*half_len),(j):(j+2*half_len),(k):(k+2*half_len)]) > tomo_std:
                    mask[i,j,k] = 1
    return mask

def stdmask_mpi(tomo,cubelen=20,cubesize=50,ncpu=20,if_rescale=True):
    from mwr.util.toTile import reform3D
    from skimage.transform import resize
    from functools import partial

    sp=np.array(tomo.shape)

    if if_rescale == True:
        sp2 = (sp/2).astype(int)
        data = resize(tomo,sp2,anti_aliasing=True)
        data=np.expand_dims(data,axis=-1)
        tomo_std = np.std(data)
    else:
        data=np.expand_dims(tomo,axis=-1)
        tomo_std = np.std(data)
    rf=reform3D(data)
    cube_ls=rf.pad_and_crop_new(cubesize=cubesize, cropsize=cubesize)
    stdmask_p = partial(stdmask,cubelen=cubelen,std=tomo_std) #use whole tomo std instead of cubes'
    from multiprocessing import Pool
    with Pool(ncpu) as p:
       mask_ls = p.map(stdmask_p,cube_ls)
    mask_ls=np.array(mask_ls)
    mask=rf.restore_from_cubes_new(mask_ls,cubesize=cubesize, cropsize=cubesize)
    if if_rescale == True:
        mask_out=np.zeros(sp)
        mask_out[1::2,1::2,1::2]=mask
    else:
        mask_out=mask
    mask_out = mask_out.astype(np.uint8)
    return mask_out

def stdmask(tomo,side=10,threshold=1):
    from scipy.signal import convolve
    tomo = tomo.astype(np.float32)
    tomosq = tomo**2
    ones = np.ones(tomo.shape)
    eps = 0.0001
    kernel = np.ones((2*side+1, 2*side+1, 2*side+1))
    s = convolve(tomo, kernel, mode="same")
    s2 = convolve(tomosq, kernel, mode="same")
    ns = convolve(ones, kernel, mode="same")+eps

    out = np.sqrt((s2 - s**2 / ns) / ns)
    out = out>np.std(tomo)*threshold
    return out.astype(np.uint8)

  
if __name__ == "__main__":
    import sys
    import mrcfile
    args = sys.argv
    with mrcfile.open(args[1]) as n:
        tomo = n.data
    mask = stdmask_mpi(tomo,cubelen=20,cubesize=80,ncpu=20,if_rescale=True)
    with mrcfile.new(args[2],overwrite=True) as n:
        n.set_data(mask)
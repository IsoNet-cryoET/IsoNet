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

def create_seed_2D(img2D,nPatchesPerSlice,patchSideLen):
    y,x = img2D.shape[0],img2D.shape[1]

    seedx = np.random.rand(nPatchesPerSlice)*(x-patchSideLen)+patchSideLen//2
    seedy = np.random.rand(nPatchesPerSlice)*(y-patchSideLen)+patchSideLen//2
    seedx = seedx.astype(int)
    seedy = seedy.astype(int)

    return seedx,seedy


def create_filter_seed_2D(img2D,nPatchesPerSlice,patchSideLen):
    pass



def crop_seed2D(img2D,seedx,seedy,cropx,cropy):
    y,x = img2D.shape[0],img2D.shape[1]

    patchshape = img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)].shape

    return img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)].astype(int)

def create_patch_image_2D(image2D,seedx,seedy,patchSideLen):
    y,x = image2D.shape
    patches = np.zeros([seedx.size,patchSideLen,patchSideLen])
    for i in range(seedx.size):
        patches[i] = crop_seed2D(image2D,seedx[i],seedy[i],patchSideLen,patchSideLen)
    return patches


def save_patch_image_2D(patches):
    imsave('patches.tif',patches)

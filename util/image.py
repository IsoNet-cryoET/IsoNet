import numpy as np



def toUint8(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*255
    data=np.clip(data,0,255)
    data=data.astype(np.uint8)
    return data

def toUint16(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*65535
    data=data.astype(np.uint16)
    return data


def crop_center(img,cropx,cropy,cropz):
    z,y,x = img.shape[0],img.shape[1],img.shape[2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

def create_seed_2D(img2D,nPatchesPerSlice,patchSideLen):
    y,x = img2D.shape[0],img2D.shape[1]

    seedx = np.random.rand(nPatchesPerSlice)*(x-patchSideLen)+patchSideLen//2
    seedy = np.random.rand(nPatchesPerSlice)*(y-patchSideLen)+patchSideLen//2
    seedx = seedx.astype(int)
    seedy = seedy.astype(int)
    return seedx,seedy

def print_filter_mask(img3D,nPatchesPerSlice,patchSideLen,threshold=0.4,percentile=99.9):
    sp=img3D.shape
    mask=np.zeros(sp).astype(np.uint8)
    myfilter = no_background_patches(threshold=threshold,percentile=percentile)
    for i in range(sp[0]):
        mask[i]=myfilter(img3D[i].reshape(1,sp[1],sp[2]),(patchSideLen,patchSideLen))

    return mask

def create_filter_seed_2D(img2D,nPatchesPerSlice,patchSideLen, patch_mask):
    
    sp=img2D.shape

    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((patchSideLen,patchSideLen), sp)])
    valid_inds = np.where(patch_mask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]),nPatchesPerSlice , replace=len(valid_inds[0])< nPatchesPerSlice)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return rand_inds[1],rand_inds[0]


def create_cube_seeds(img3D,nCubesPerImg,cubeSideLen,mask=None):
    sp=img3D.shape
    if mask is None:
        cubeMask=np.ones(sp)
    else:
        cubeMask=mask
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((cubeSideLen,cubeSideLen,cubeSideLen), sp)])
    valid_inds = np.where(cubeMask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds = np.random.choice(len(valid_inds[0]), nCubesPerImg, replace=len(valid_inds[0]) < nCubesPerImg)
    rand_inds = [v[sample_inds] for v in valid_inds]
    return (rand_inds[0],rand_inds[1], rand_inds[2])


def crop_seed2D(img2D,seedx,seedy,cropx,cropy):
    y,x = img2D.shape[0],img2D.shape[1]
    
    patchshape = img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)].shape
    return img2D[seedy-(cropy//2):seedy+(cropy)//2,seedx-(cropx//2):seedx+(cropx//2)]#.astype(int)

def create_patch_image_2D(image2D,seedx,seedy,patchSideLen):
    y,x = image2D.shape
    patches = np.zeros([seedx.size,patchSideLen,patchSideLen])
    for i in range(seedx.size):
        patches[i] = crop_seed2D(image2D,seedx[i],seedy[i],patchSideLen,patchSideLen)
    return patches
def crop_cubes(img3D,seeds,cubeSideLen):
    size=len(seeds[0])
    cube_size=(cubeSideLen,cubeSideLen,cubeSideLen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,cube_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes

def rotate(data,angle,axes=0):
    sp=data.shape
    theta=angle/180*np.pi
    cos_theta=np.cos(theta)
    sin_theta=np.sin(theta)
    sideLen=np.min([sp[1],sp[2]])//np.sqrt(2)
    sideLen=sideLen.astype(np.uint16)
    rotated=np.zeros([sp[0],sideLen,sideLen],dtype=np.uint8)
    for _z in range(sp[0]):
        print(_z)
        for _y in range(sideLen):
            for _x in range(sideLen):
                y_prime=int((_y-sideLen//2)*cos_theta-(_x-sideLen//2)*sin_theta+sp[1]//2)
                x_prime= int((_x-sideLen//2)*cos_theta+(_y-sideLen//2)*sin_theta+sp[2]//2)
                rotated[_z,_y,_x]=data[_z,y_prime,x_prime]
    return rotated





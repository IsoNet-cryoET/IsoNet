import numpy as np
import mrcfile

def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out


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

def crop_cubes(img3D,seeds,cubeSideLen):
    size=len(seeds[0])
    cube_size=(cubeSideLen,cubeSideLen,cubeSideLen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,cube_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes

class DataCubes:

    def __init__(self, tomogram, nCubesPerImg=32, cubeSideLen=32, cropsize=32, mask = None, validationSplit=0.1, noise_folder = None, noise_level = 0.5):

        self.tomogram = tomogram
        self.nCubesPerImg = nCubesPerImg
        self.cubeSideLen = cubeSideLen
        self.cropsize = cropsize
        self.mask = mask
        self.validationSplit = validationSplit
        self.__cubesY_padded = None
        self.__cubesY = None
        self.__cubesX = None
        self.noise_folder = noise_folder
        self.noise_level = noise_level


    @property
    def cubesY_padded(self):
        if self.__cubesY_padded is None:
            seeds=create_cube_seeds(self.tomogram,self.nCubesPerImg,self.cropsize,self.mask)
            self.__cubesY_padded=crop_cubes(self.tomogram,seeds,self.cropsize)
            self.__cubesY_padded = self.__cubesY_padded.astype(np.float32)
        return self.__cubesY_padded

    @property
    def cubesY(self):
        if self.__cubesY is None:
            self.__cubesY = self.crop_to_size(self.cubesY_padded, self.cubeSideLen)
        return self.__cubesY

    @property
    def cubesX(self):
        if self.__cubesX is None:
            #print('here', self.tomogram)
            from IsoNet.simulation.simulate import apply_wedge

            res = list(map(apply_wedge, self.cubesY_padded))

            cubesX_padded = np.array(res, dtype = np.float32)
            self.__cubesX = self.crop_to_size(cubesX_padded, self.cubeSideLen)

            if self.noise_folder is not None:
                import os
                path_noise = sorted([self.noise_folder+'/'+f for f in os.listdir(self.noise_folder)])
                path_index = np.random.permutation(len(path_noise))[0:self.__cubesX.shape[0]]
                def read_vol(f):
                    with mrcfile.open(f) as mf:
                        res = mf.data
                    return res
                noise_volume = np.array([read_vol(path_noise[j]) for j in path_index])
                self.__cubesX += self.noise_level * noise_volume / np.std(noise_volume)

        return self.__cubesX


    def crop_to_size(self, array, size):
        start = self.cropsize//2 - size//2
        end = self.cropsize//2 + size//2
        return array[:,start:end,start:end,start:end]

    def create_training_data3D(self):
        n_val = int(self.cubesX.shape[0]*self.validationSplit)
        n_train = int(self.cubesX.shape[0])-n_val
        X_train, Y_train = self.cubesX[:n_train], self.cubesY[:n_train]
        X_train, Y_train = np.expand_dims(X_train,-1), np.expand_dims(Y_train,-1)
        X_test, Y_test = self.cubesX[-n_val:], self.cubesY[-n_val:]
        X_test, Y_test = np.expand_dims(X_test,-1), np.expand_dims(Y_test,-1)
        return (X_train, Y_train),(X_test, Y_test)


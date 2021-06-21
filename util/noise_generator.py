#!/usr/bin/env python3
import numpy as np
import random
from scipy.ndimage import rotate
from IsoNet.preprocessing.simulate import apply_wedge
import os
import mrcfile
from skimage.transform import iradon
from IsoNet.util.utils import mkfolder

def simulate_noise1(params):
    np.random.seed(random.randint(0,100000))
    gs_cube = np.random.normal(size = (params[0],)*3).astype(np.float32)
    gs_wedged = apply_wedge(gs_cube,ld1=1,ld2=0)
    return gs_wedged

def simulate_noise2(params):
    def rt(inp):
        np.random.seed(random.randint(0,100000))
        (angle,size) = inp
        b = np.random.normal(size=(size,int(size*1.4))).astype(np.float32)
        b = np.repeat(b[np.newaxis, :, : ], int(size*1.4), axis=0)
        b = rotate(b,angle,(0,2), reshape=False, order = 1)
        return b
    a = np.arange(params[1],params[2]+params[3],params[3])
    res = list(map(rt, zip(a,[params[0]]*len(a))))
    res = np.average(np.array(res), axis = 0)
    start = int(params[0]*0.2)
    res = res[start:start+params[0],:,start:start+params[0]]
    return res

def make_noise(output_folder, number_volume, cubesize=64, minangle=-60,maxangle=60, anglestep=2, start=0,ncpus=25, mode=1):
    if mode==1:
        noise_func = simulate_noise1
    else:
        noise_func = simulate_noise2
    params = [cubesize, minangle, maxangle, anglestep]
    try:
        os.makedirs(output_folder)
    except OSError:
        print ("  " )

    count = 0
    for count in range(0,number_volume, ncpus):
        print(count+start)
        from multiprocessing import Pool
        with Pool(ncpus) as p:
            res = p.map(noise_func, [params]*ncpus)
        res = list(res)
        for i,img in enumerate(res):
            with mrcfile.new('{}/n_{:0>5d}.mrc'.format(output_folder,count+i+start), overwrite=True) as output_mrc:
                output_mrc.set_data(img)

def make_noise_one(cubesize=64, minangle=-60,maxangle=60, anglestep=2, mode=1):
    if mode==1:
        noise_func = simulate_noise1 # gaussian noise + missing-wedge
    else:
        noise_func = simulate_noise2 # back-projection
    params = [cubesize, minangle, maxangle, anglestep]
    simu_noise = noise_func(params)
    return simu_noise


##### Another Strategy for generating noise: 
# 1. Generate a large noise volume at the first iteration that the training needs noises
# 2. Crop out thousands noise volumes and save them in ./$result_dir/training_noise
# 3. In Later iterations, these volumes will be re-used and no need to generate any more.
class NoiseMap:
    noise_map = None

    @staticmethod
    def refresh(size_big, filter = 'ramp' , ncpus = 1):
        NoiseMap.noise_map = simulate_noise([size_big, filter, ncpus])

    @staticmethod
    def get_one(size):
        shp = NoiseMap.noise_map.shape
        start = np.random.randint(0, shp[0]-size, 3)
        return NoiseMap.noise_map[start[0]:start[0]+size, start[1]:start[1]+size,start[2]:start[2]+size]

#filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
angles = np.arange(-60,62,3)

def part_iradon_ramp(x):
    return iradon(x, angles, filter_name = 'ramp' )

def part_iradon_hamming(x):
    return iradon(x, angles, filter_name = 'hamming' )

def part_iradon_shepp(x):
    return iradon(x, angles, filter_name = 'shepp-logan' )

def part_iradon_cosine(x):
    return iradon(x, angles, filter_name = 'cosine' )

def part_iradon_nofilter(x):
    return iradon(x, angles, filter_name = None)

def simulate_noise(params):
    size = params[0]
    sinograms = np.random.normal(size=(size,int(size*1.4),len(angles)))
    start=int(params[0]*0.2)
    from multiprocessing import Pool
    with Pool(params[2]) as p:
        if params[1] == 'ramp':
            res = p.map(part_iradon_ramp,sinograms)
        elif params[1] == 'hamming':
            res = p.map(part_iradon_hamming,sinograms)
        else:
            res = p.map(part_iradon_nofilter,sinograms)
            
        
    iradon_image = np.rot90(np.array(list(res), dtype=np.float32)[:,start:start+params[0],start:start+params[0]], k = 1 , axes = (0,1))
    return iradon_image

def make_noise_folder(noise_folder,noise_filter,cube_size,num_noise=1000,ncpus=1,large_side=1000):
    mkfolder(noise_folder)
    print('generating large noise volume; mode: {}'.format(noise_filter))
    NoiseMap.refresh(large_side, noise_filter, ncpus)
                        
    for i in range(num_noise):
        img = NoiseMap.get_one(cube_size)
        with mrcfile.new('{}/n_{:0>5d}.mrc'.format(noise_folder,i), overwrite=True) as output_mrc:
            output_mrc.set_data(img)

if __name__ == '__main__':
    import mrcfile
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('output_folder', type=str, default=None, help='output folder')
    parser.add_argument('number_volume', type=int, default=100, help='number of output mrc file')
    parser.add_argument('--cubesize', type=int, default=64, help='size of cube')
    parser.add_argument('--minangle', type=float, default=-60, help='')
    parser.add_argument('--maxangle', type=float, default=60, help='')
    parser.add_argument('--anglestep', type=float, default=2, help='')
    parser.add_argument('--start', type=int, default=1, help='name the volume with start number')
    parser.add_argument('--ncpus', type=int, default=8, help='number of cpus')
    args = parser.parse_args()
    params = [args.cubesize, args.minangle, args.maxangle, args.anglestep]
    try:
        os.makedirs(args.output_folder)
    except OSError:
        print ("  " )

    count = 0
    for count in range(0,args.number_volume, args.ncpus):
        print(count+args.start)
        from multiprocessing import Pool
        with Pool(args.ncpus) as p:
            res = p.map(noise_func, [params]*args.ncpus)
        res = list(res)
        for i,img in enumerate(res):
            with mrcfile.new('{}/n_{:0>5d}.mrc'.format(args.output_folder,count+i+args.start), overwrite=True) as output_mrc:
                output_mrc.set_data(img)


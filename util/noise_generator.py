#!/usr/bin/env python3
import numpy as np
import random
from scipy.ndimage import rotate
from IsoNet.preprocessing.simulate import apply_wedge
import os
import mrcfile

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
        noise_func = simulate_noise1
    else:
        noise_func = simulate_noise2
    params = [cubesize, minangle, maxangle, anglestep]
    simu_noise = noise_func(params)
    return simu_noise


# if __name__ == "__main__":
#     import time
#     import mrcfile
#     start = time.time()
#     a = make_noise_one(mode=1)
#     end = time.time()
#     print('Time consuming:',end-start)

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


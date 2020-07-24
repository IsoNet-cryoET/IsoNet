#!/usr/bin/env python3

import sys
import mrcfile
args = sys.argv
from mwr.util.filter import stdmask_mpi,maxmask,stdmask
import numpy as np

def make_mask(tomo_path, mask_name,side = 8,percentile=30,threshold=1):
    from skimage.transform import resize
    with mrcfile.open(tomo_path) as n:
        tomo = n.data
    sp=np.array(tomo.shape)
    sp2 = (sp/2).astype(int)
    bintomo = resize(tomo,sp2,anti_aliasing=True)
    mask1 = maxmask(bintomo,side=side, percentile=percentile)
    mask2 = stdmask(bintomo,side=side,threshold=threshold)
    out_mask = np.multiply(mask1,mask2)
    out = resize(out_mask.astype(np.float32),sp,anti_aliasing=True)
    out = out>0.5
    with mrcfile.new(mask_name,overwrite=True) as n:
        n.set_data(out.astype(np.uint8))
    # with mrcfile.new('./test_mask1.rec',overwrite=True) as n:
    #     n.set_data(mask1.astype(np.float32))    
    # with mrcfile.new('./test_mask2.rec',overwrite=True) as n:
    #     n.set_data(mask2.astype(np.float32))
if __name__ == "__main__":
# the first arg is tomo name the second is mask name
    make_mask(args[1],args[2])


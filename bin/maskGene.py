#!/usr/bin/env python3
import sys
import mrcfile
args = sys.argv
from mwr.util.filter import stdmask_mpi
if __name__ == "__main__":

    with mrcfile.open(args[1]) as n:
        tomo = n.data
    mask = stdmask_mpi(tomo,cubelen=20,cubesize=80,ncpu=20,if_rescale=True)
    with mrcfile.new(args[2],overwrite=True) as n:
        n.set_data(mask)
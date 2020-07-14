# Copyright (c) 2016, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.
"""
dtypes
------

numpy dtypes used by the ``mrcfile.py`` library.

The dtypes are defined in a separate module because they do not interact nicely
with the ``from __future__ import unicode_literals`` feature used in the rest
of the package.

"""

# Import Python 3 features for future-proofing
# Deliberately do NOT import unicode_literals due to a bug in numpy dtypes:
# https://github.com/numpy/numpy/issues/2407
from __future__ import absolute_import, division, print_function

import numpy as np


HEADER_DTYPE = np.dtype([
    ('nx', 'i4'),          # Number of columns
    ('ny', 'i4'),          # Number of rows
    ('nz', 'i4'),          # Number of sections
    
    ('mode', 'i4'),        # Mode; indicates type of values stored in data block
    
    ('nxstart', 'i4'),     # Starting point of sub-image
    ('nystart', 'i4'),
    ('nzstart', 'i4'),
    
    ('mx', 'i4'),          # Grid size in X, Y and Z
    ('my', 'i4'),
    ('mz', 'i4'),
    
    ('cella', [            # Cell size in angstroms
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4')
    ]),
    
    ('cellb', [            # Cell angles in degrees
        ('alpha', 'f4'),
        ('beta', 'f4'),
        ('gamma', 'f4')
    ]),
    
    ('mapc', 'i4'),        # map column  1=x,2=y,3=z.
    ('mapr', 'i4'),        # map row     1=x,2=y,3=z.
    ('maps', 'i4'),        # map section 1=x,2=y,3=z.
    
    ('dmin', 'f4'),        # Minimum pixel value
    ('dmax', 'f4'),        # Maximum pixel value
    ('dmean', 'f4'),       # Mean pixel value
    
    ('ispg', 'i4'),        # space group number
    ('nsymbt', 'i4'),      # number of bytes in extended header
    
    ('extra1', 'V8'),      # extra space, usage varies by application
    ('exttyp', 'S4'),      # code for the type of extended header
    ('nversion', 'i4'),    # version of the MRC format
    ('extra2', 'V84'),     # extra space, usage varies by application
    
    ('origin', [           # Origin of image
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4')
    ]),
    
    ('map', 'S4'),         # Contains 'MAP ' to identify file type
    ('machst', 'u1', 4),   # Machine stamp; identifies byte order
    
    ('rms', 'f4'),         # RMS deviation of densities from mean density
    
    ('nlabl', 'i4'),       # Number of labels with useful data
    ('label', 'S80', 10)   # 10 labels of 80 characters
])


VOXEL_SIZE_DTYPE = np.dtype([
    ('x', 'f4'),
    ('y', 'f4'),
    ('z', 'f4')
])


# FEI extended header dtype for metadata version 0, as described in the EPU
# manual. Note that the FEI documentation is unclear about the endianness of
# the data in the extended header. Probably, it is always little-endian, and
# therefore the data might be misinterpreted on a big-endian machine, but this
# has not been tested.
FEI_EXTENDED_HEADER_DTYPE = np.dtype([
    ('Metadata size', 'i4'),
    ('Metadata version', 'i4'),
    ('Bitmask 1', 'u4'),
    ('Timestamp', 'f8'),  # Not specified, but suspect this is in days after 1/1/1900
    ('Microscope type', 'S16'),
    ('D-Number', 'S16'),
    ('Application', 'S16'),
    ('Application version', 'S16'),
    ('HT', 'f8'),
    ('Dose', 'f8'),
    ('Alpha tilt', 'f8'),
    ('Beta tilt', 'f8'),
    ('X-Stage', 'f8'),
    ('Y-Stage', 'f8'),
    ('Z-Stage', 'f8'),
    ('Tilt axis angle', 'f8'),
    ('Dual axis rotation', 'f8'),
    ('Pixel size X', 'f8'),
    ('Pixel size Y', 'f8'),
    ('Unused range', 'S48'),
    ('Defocus', 'f8'),
    ('STEM Defocus', 'f8'),
    ('Applied defocus', 'f8'),
    ('Instrument mode', 'i4'),
    ('Projection mode', 'i4'),
    ('Objective lens mode', 'S16'),
    ('High magnification mode', 'S16'),
    ('Probe mode', 'i4'),
    ('EFTEM On', '?'),
    ('Magnification', 'f8'),
    ('Bitmask 2', 'u4'),
    ('Camera length', 'f8'),
    ('Spot index', 'i4'),
    ('Illuminated area', 'f8'),
    ('Intensity', 'f8'),
    ('Convergence angle', 'f8'),
    ('Illumination mode', 'S16'),
    ('Wide convergence angle range', '?'),
    ('Slit inserted', '?'),
    ('Slit width', 'f8'),
    ('Acceleration voltage offset', 'f8'),
    ('Drift tube voltage', 'f8'),
    ('Energy shift', 'f8'),
    ('Shift offset X', 'f8'),
    ('Shift offset Y', 'f8'),
    ('Shift X', 'f8'),
    ('Shift Y', 'f8'),
    ('Integration time', 'f8'),
    ('Binning Width', 'i4'),
    ('Binning Height', 'i4'),
    ('Camera name', 'S16'),
    ('Readout area left', 'i4'),
    ('Readout area top', 'i4'),
    ('Readout area right', 'i4'),
    ('Readout area bottom', 'i4'),
    ('Ceta noise reduction', '?'),
    ('Ceta frames summed', 'i4'),
    ('Direct detector electron counting', '?'),
    ('Direct detector align frames', '?'),
    ('Camera param reserved 0', 'i4'),
    ('Camera param reserved 1', 'i4'),
    ('Camera param reserved 2', 'i4'),
    ('Camera param reserved 3', 'i4'),
    ('Bitmask 3', 'u4'),
    ('Camera param reserved 4', 'i4'),
    ('Camera param reserved 5', 'i4'),
    ('Camera param reserved 6', 'i4'),
    ('Camera param reserved 7', 'i4'),
    ('Camera param reserved 8', 'i4'),
    ('Camera param reserved 9', 'i4'),
    ('Phase Plate', '?'),
    ('STEM Detector name', 'S16'),
    ('Gain', 'f8'),
    ('Offset', 'f8'),
    ('STEM param reserved 0', 'i4'),
    ('STEM param reserved 1', 'i4'),
    ('STEM param reserved 2', 'i4'),
    ('STEM param reserved 3', 'i4'),
    ('STEM param reserved 4', 'i4'),
    ('Dwell time', 'f8'),
    ('Frame time', 'f8'),
    ('Scan size left', 'i4'),
    ('Scan size top', 'i4'),
    ('Scan size right', 'i4'),
    ('Scan size bottom', 'i4'),
    ('Full scan FOV X', 'f8'),
    ('Full scan FOV Y', 'f8'),
    ('Element', 'S16'),
    ('Energy interval lower', 'f8'),
    ('Energy interval higher', 'f8'),
    ('Method', 'i4'),
    ('Is dose fraction', '?'),
    ('Fraction number', 'i4'),
    ('Start frame', 'i4'),
    ('End frame', 'i4'),
    ('Input stack filename', 'S80'),
    ('Bitmask 4', 'u4'),
    ('Alpha tilt min', 'f8'),
    ('Alpha tilt max', 'f8')
])

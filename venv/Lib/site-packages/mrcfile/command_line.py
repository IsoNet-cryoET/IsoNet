# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

"""
command_line
------------

Module for functions used as command line entry points.

The names of the corresponding command line scripts can be found in the
``entry_points`` section of ``setup.py``.

"""

# Import Python 3 features for future-proofing
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

import mrcfile


def print_headers(names=None, print_file=None):
    """
    Print the MRC header contents from a list of files.
    
    This function opens files in permissive mode to allow headers of invalid
    files to be examined.
    
    Args:
        names: A list of file names. If not given or :data:`None`, the names
            are taken from the command line arguments.
        print_file: The output text stream to use for printing the headers.
            This is passed directly to the ``print_file`` argument of
            :meth:`~mrcfile.mrcobject.MrcObject.print_header`. The default is
            :data:`None`, which means output will be printed to
            :data:`sys.stdout`.
    """
    if names is None:
        names = sys.argv[1:]
    for name in names:
        with mrcfile.open(name, permissive=True, header_only=True) as mrc:
            mrc.print_header(print_file=print_file)

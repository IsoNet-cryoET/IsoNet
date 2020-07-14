# Copyright (c) 2018, Science and Technology Facilities Council
# This software is distributed under a BSD licence. See LICENSE.txt.

"""
validator
---------

Module for top-level functions that validate MRC files.

This module is runnable to allow files to be validated easily from the command
line.

"""

# Import Python 3 features for future-proofing
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys

from . import load_functions


def main():
    """
    Validate a list of MRC files given as command arguments.
    
    The return value is used as the process exit code when this function is
    called by running this module or from the corresponding ``console_scripts``
    entry point.
    
    Returns:
        ``0`` if all command arguments are names of valid MRC files. ``1`` if
        no file names are given or any of the files is not a valid MRC file.
    """
    names = sys.argv[1:]
    if len(names) > 0:
        if validate_all(names):
            return 0
    else:
        print("Usage: python -m mrcfile.validator FILE...")
    return 1


def validate_all(names, print_file=None):
    """Validate a list of MRC files.
    
    This function calls :func:`validate` for each file name in the given list.
    
    Note that :data:`False` will be returned immediately if a file is found to
    be invalid. Subsequent files will not be checked.
    
    Args:
        names: A sequence of file names to open and validate.
        print_file: The output text stream to use for printing messages about
            the validation. This is passed directly to the ``print_file``
            argument of the :func:`validate` function. The default is
            :data:`None`, which means output will be printed to
            :data:`sys.stdout`.
    
    Returns:
        :data:`True` if all of the files are valid, or :data:`False` if any of
        the files do not meet the MRC format specification in any way.
    
    Raises:
        :exc:`OSError`: If one of the files does not exist or cannot be opened.
    
    Warns:
        RuntimeWarning: If one of the files is seriously invalid because it has
            no map ID string, an incorrect machine stamp, an unknown mode
            number, or is not the same size as expected from the header.
    """
    return all(validate(name, print_file) for name in names)


def validate(name, print_file=None):
    """Validate an MRC file.
    
    This function first opens the file by calling :func:`open` (with
    ``permissive=True``), then calls :meth:`~mrcfile.mrcfile.MrcFile.validate`,
    which runs a series of tests to check whether the file complies with the
    MRC2014 format specification.
    
    If the file is completely valid, this function returns :data:`True`,
    otherwise it returns :data:`False`. Messages explaining the validation
    result will be printed to :data:`sys.stdout` by default, but if a text
    stream is given (using the ``print_file`` argument) output will be printed
    to that instead.
    
    Badly invalid files will also cause :mod:`warning <warnings>` messages to
    be issued, which will be written to :data:`sys.stderr` by default. See the
    documentation of the :mod:`warnings` module for information on how to
    suppress or capture warning output.
    
    Because the file is opened by calling :func:`open`, gzip- and
    bzip2-compressed MRC files can be validated easily using this function.
    
    After the file has been opened, it is checked for problems. The tests are:
    
    #. MRC format ID string: The ``map`` field in the header should contain
       "MAP ".
    #. Machine stamp: The machine stamp should contain one of
       ``0x44 0x44 0x00 0x00``, ``0x44 0x41 0x00 0x00`` or
       ``0x11 0x11 0x00 0x00``.
    #. MRC mode: the ``mode`` field should be one of the supported mode
       numbers: 0, 1, 2, 4 or 6.
    #. Map and cell dimensions: The header fields ``nx``, ``ny``, ``nz``,
       ``mx``, ``my``, ``mz``, ``cella.x``, ``cella.y`` and ``cella.z`` must
       all be positive numbers.
    #. Axis mapping: Header fields ``mapc``, ``mapr`` and ``maps`` must contain
       the values 1, 2, and 3 (in any order).
    #. Volume stack dimensions: If the spacegroup is in the range 401--630,
       representing a volume stack, the ``nz`` field should be exactly
       divisible by ``mz`` to represent the number of volumes in the stack.
    #. Header labels: The ``nlabl`` field should be set to indicate the number
       of labels in use, and the labels in use should appear first in the label
       array.
    #. MRC format version: The ``nversion`` field should be 20140 for
       compliance with the MRC2014 standard.
    #. Extended header type: If an extended header is present, the ``exttyp``
       field should be set to indicate the type of extended header.
    #. Data statistics: The statistics in the header should be correct for the
       actual data in the file, or marked as undetermined.
    #. File size: The size of the file on disk should match the expected size
       calculated from the MRC header.
    
    Args:
        name: The file name to open and validate.
        print_file: The output text stream to use for printing messages about
            the validation. This is passed directly to the ``file`` argument of
            Python's :func:`print` function. The default is :data:`None`, which
            means output will be printed to :data:`sys.stdout`.
    
    Returns:
        :data:`True` if the file is valid, or :data:`False` if the file does
        not meet the MRC format specification in any way.
    
    Raises:
        :exc:`OSError`: If the file does not exist or cannot be opened.
    
    Warns:
        RuntimeWarning: If the file is seriously invalid because it has no map
            ID string, an incorrect machine stamp, an unknown mode number, or
            is not the same size as expected from the header.
    """
    with load_functions.open(name, permissive=True) as mrc:
        return mrc.validate(print_file=print_file)


if __name__ == '__main__':
    sys.exit(main())

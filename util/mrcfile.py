#!/usr/bin/env python
# encoding: utf-8
"""Created on Sep.23 2016

"""

def readMrcNumpy(mrcFile):
    import struct
    import numpy as np
    fp = open(mrcFile, 'rb')
    header=struct.unpack("10i6f3i3f2i25f3f4s4sfi800s", fp.read(1024))

    if header[3] == 0:
        data = np.fromfile(fp, dtype=np.uint8)+128
    elif header[3] == 1:
        data = np.fromfile(fp, dtype=np.uint16)
    elif header[3] == 2:
        data = np.fromfile(fp, dtype=np.int16)
    elif header[3] == 3:
        data = np.fromfile(fp, dtype = np.float)
    elif header[3] == 6:
        data = np.fromfile(fp, dtype = np.uint16)
    fp.close()
    data=data.reshape(header[0:3][2],header[0:3][1],header[0:3][0])
    return data

#def saveMrcNumpy(mrcFile, array):

import struct
headerStruct = struct.Struct("3ii3i3i3f3f3i3fii25i3f4c4sfi80s80s80s80s80s80s80s80s80s80s")

def readHeader(fileName):
    header = open(fileName, "rb").read(1024)
    header = headerStruct.unpack(header)

def writeHeader(fileName, header):
    f = open(fileName, "wb")
    f.write(header)
    return f

def writeMrc(fileName, array):
    import numpy as np
    f = open(fileName, 'wb')#writeHeader(fileName, header)
    #array = array.astype(np.uint8)
    f.write(array.tostring())
    f.close()

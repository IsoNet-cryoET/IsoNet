#!/usr/bin/env python
# encoding: utf-8

from keras.models import load_model
from tifffile import imread
new_model = load_model('weights_last.h5')
img = imread('p190-bin8-1.tif')
shape=list(img.shape)
shape.append(1)
new_model.predict(img.reshape(shape),batch_size=128)

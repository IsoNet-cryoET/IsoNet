from tensorflow.keras.layers import Dropout, Activation, BatchNormalization,LeakyReLU
from tensorflow.keras.layers import Add, Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, AveragePooling2D, AveragePooling3D, Conv3DTranspose
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras import regularizers
import mrcfile
import os
import sys

def encoder_block(layer_in, n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True, activation='relu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv3D(n_filters, kernel, strides=strides, padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        # g = BatchNormalization()(g, training=True)
        g = BatchNormalization()(g)
    if dropout is not None and dropout>0:
        # g=Dropout(dropout)(g, training=True)
        g=Dropout(dropout)(g)
    g = LeakyReLU(alpha=0.05)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True,activation='relu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv3DTranspose(n_filters, kernel, strides=strides, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    if batchnorm:
        # g = BatchNormalization()(g, training=True)
        g = BatchNormalization()(g)
    # conditionally add dropout
    if dropout is not None and dropout>0:
        # g = Dropout(dropout)(g, training=True)
        g=Dropout(dropout)(g)
    # merge with skip connection
    if skip_in is not None:
        g = Concatenate()([g, skip_in])
    # relu activation
    g = LeakyReLU(alpha=0.05)(g)
    return g

def encoder_layer(layer_in,depth,settings):
    # e = define_encoder_block(layer_in,settings.filter_base*2**n,strides=(2,2,2))(layer_in)
    e = layer_in
    for i in range(settings.nconvs_per):
        e = encoder_block(e, settings.filter_base*2**depth, strides=(1,1,1))
    return e   

def build_unet(settings):
    def _func(inp):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        # encoder model
        # every layer out has a stride(2,2,2) conv layer at the end;
        l1_out = encoder_layer(inp, 1,settings)
        l1_downs = encoder_block(l1_out, settings.filter_base*2**1, strides=(2,2,2),dropout=None,batchnorm=False,activation='linear')
        l2_out = encoder_layer(l1_downs,2,settings)
        l2_downs = encoder_block(l2_out, settings.filter_base*2**2, strides=(2,2,2),dropout=None,batchnorm=False,activation='linear')
        l3_out = encoder_layer(l2_downs,3,settings)
        l3_downs = encoder_block(l3_out, settings.filter_base*2**3, strides=(2,2,2),dropout=None,batchnorm=False,activation='linear')
        # bottleneck, no batch norm and relu
        b = Conv3D(settings.filter_base*2**4, (3,3,3), strides=(1,1,1), padding='same', kernel_initializer=init)(l3_downs)
        b = LeakyReLU(alpha=0.05)(b)
        b = Conv3D(settings.filter_base*2**3, (3,3,3), strides=(1,1,1), padding='same', kernel_initializer=init)(b)
        # decoder model
        d3_ups = decoder_block(b, l3_out, settings.filter_base*2**3, dropout=None,batchnorm=False,activation='linear')
        d3_out = encoder_layer(d3_ups,3,settings)
        d2_ups = decoder_block(d3_out,l2_out,settings.filter_base*2**2,dropout=None,batchnorm=False,activation='linear')
        d2_out = encoder_layer(d2_ups,2,settings)
        d1_ups = decoder_block(d2_out,l1_out,settings.filter_base*2**1,dropout=None,batchnorm=False,activation='linear')
        d1_out = encoder_layer(d1_ups,1,settings)
        final = Conv3D(1,(1,1,1), strides=(1,1,1), padding='same', kernel_initializer=init)(d1_out)
        return final
    return _func
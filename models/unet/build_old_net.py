# unet from mwr yuntao7.28 version
import tensorflow
from tensorflow.keras.layers import Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Add, Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import numpy as np
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import regularizers

def conv_blocks(n_filter, kernel,
                activation="relu",
                padding="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):
    def _func(last_layer):
        #print(last_layer.shape)
        if len(kernel)==2:
            conv=Conv2D(n_filter,kernel,
                        padding=padding,
                        kernel_initializer=init,**kwargs)(last_layer)
        elif len(kernel)==3:
            conv=Conv3D(n_filter,kernel,
                        padding=padding,
                        kernel_initializer=init,**kwargs)(last_layer)
        if batch_norm:
            conv=BatchNormalization()(conv)
        conv=Activation(activation)(conv)
        if dropout is not None and dropout>0:
            conv=Dropout(dropout)(conv)

        return conv
    return _func

def unet_block(filter_base=32,depth=2,convs_per_depth=2,
               kernel=(3,3),
               batch_norm=False,
               dropout=0.0,
               pool=(2,2)):
    if len(kernel)==2:
        MaxPooling=MaxPooling2D
        UpSampling=UpSampling2D
    else:
        MaxPooling=MaxPooling3D
        UpSampling=UpSampling3D
    def _func(inputs):
        concatenate = []
        layer = inputs
        for n in range(depth):
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,name="down_level_%s_no_%s" % (n, i))(layer)
            concatenate.append(layer)
            layer = MaxPooling(pool_size=pool)(layer)

        for i in range(convs_per_depth -1):
            layer = conv_blocks(filter_base*2**depth,kernel,
                                dropout=dropout,batch_norm = batch_norm,name="middle_%s" % i)(layer)
        layer = conv_blocks(filter_base*2**max(0,depth-1),kernel,dropout=dropout,
                            batch_norm=batch_norm,name="middle_%s" % convs_per_depth)(layer)

        for n in reversed(range(depth)):
            # print('depth is : %d' % n)

            layer = Concatenate(axis=-1)([UpSampling(pool)(layer),concatenate[n]])
            for i in range(convs_per_depth-1):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i))(layer)
            layer = conv_blocks(filter_base * 2 ** max(0,(n-1)), kernel, dropout=dropout,
                                batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, convs_per_depth))(layer)
        final = conv_blocks(1, (1,1,1), dropout=None,activation='linear',
                                    batch_norm=None,name="fullconv_out")(layer)
        return final
    return _func
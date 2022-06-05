from tensorflow.keras.layers import Dropout, Activation, BatchNormalization, Conv2D, Conv3D, LeakyReLU,Conv3DTranspose,Concatenate
from tensorflow.keras.initializers import RandomNormal

def conv_blocks(n_filter, kernel=(3,3,3),
                activation="relu",
                padding="same",
                dropout=0.0,
                strides=(1,1,1),
                batch_norm=False,
                **kwargs):
    def layer(last_layer):
        init = RandomNormal(stddev=0.02)
        if len(kernel)==2:
            conv=Conv2D(n_filter,kernel,
                        padding=padding,
                        strides = strides,
                        kernel_initializer=init,**kwargs)(last_layer)
        elif len(kernel)==3:
            conv=Conv3D(n_filter,kernel,
                        padding=padding,
                        strides = strides,
                        kernel_initializer=init,**kwargs)(last_layer)
        if batch_norm:
            conv=BatchNormalization()(conv)
        if dropout is not None and dropout>0:
            conv=Dropout(dropout)(conv)
        if activation is not None:
            conv = activation_my(activation)(conv)
        return conv
    return layer

def activation_my(type):
    def layer_func(layer_in):
        if type == "LeakyReLU":
            conv = LeakyReLU(alpha=0.05)(layer_in)
        else:
            conv = Activation(type)(layer_in)
        return conv
    return layer_func


def decoder_block(layer_in, skip_in , n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True,activation='relu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv3DTranspose(n_filters, kernel, strides=strides, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    if batchnorm:
        g = BatchNormalization()(g)
    # conditionally add dropout
    if dropout is not None and dropout>0:
        g = Dropout(dropout)(g)

    # if activation is not None:
    #     g = activation_my(activation)(g)
    # merge with skip connection
    if skip_in is not None:
        g = Concatenate()([g, skip_in])

    # changed activation position from line 52 to here
    if activation is not None:
        g = activation_my(activation)(g)

    return g

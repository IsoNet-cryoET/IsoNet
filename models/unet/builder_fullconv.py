from mwr.models.unet.blocks import conv_blocks
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, AveragePooling3D,Conv2D,Conv2DTranspose,Conv3D,Conv3DTranspose,Dropout,BatchNormalization,Activation,LeakyReLU
from tensorflow.keras.layers import Concatenate

def encoder_block(layer_in, n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True, activation='relu'):
    # weight initialization
    init = "glorot_uniform"
    # add downsampling layer
    g = Conv3D(n_filters, kernel, strides=strides, padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    if dropout is not None and dropout>0:
        g=Dropout(dropout)(g, training=True)
    g = LeakyReLU(alpha=0.05)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True,activation='relu'):
    # weight initialization
    init = "glorot_uniform"
    # add upsampling layer
    g = Conv3DTranspose(n_filters, kernel, strides=strides, padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout is not None and dropout>0:
        g = Dropout(dropout)(g, training=True)
    # merge with skip connection
    if skip_in is not None:
        g = Concatenate()([g, skip_in])
    # relu activation
    g = LeakyReLU(alpha=0.05)(g)
    return g

def build_unet(filter_base=32,depth=2,convs_per_depth=2,
               kernel=(3,3),
               batch_norm=False,
               dropout=0.0,
               pool=(2,2)):

    # TODO: defaut is 3D, consider 2D
    def _func(inputs):
        concatenate = []
        layer = inputs
        for n in range(depth):
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,name="down_level_%s_no_%s" % (n, i))(layer)
            concatenate.append(layer)
            # TODO: stride (2,2) for 2D case
            layer = encoder_block(layer, filter_base*2**n, strides=(2,2,2),dropout=dropout,batchnorm=False,activation='linear')

        # for i in range(convs_per_depth -1):
        #     layer = conv_blocks(filter_base*2**depth,kernel,
        #                         dropout=dropout,batch_norm = batch_norm,name="middle_%s" % i)(layer)
        # layer = conv_blocks(filter_base*2**max(0,depth-1),kernel,dropout=dropout,
        #                     batch_norm=batch_norm,name="middle_%s" % convs_per_depth)(layer)

        b = Conv3D(filter_base*2**depth, (3,3,3), strides=(1,1,1), padding='same', kernel_initializer="glorot_uniform")(layer)
        b = LeakyReLU(alpha=0.05)(b)
        layer = Conv3D(filter_base*2**(depth-1), (3,3,3), strides=(1,1,1), padding='same', kernel_initializer="glorot_uniform")(b)

        for n in reversed(range(depth)):
            # layer = Concatenate(axis=-1)([UpSampling(pool)(layer),concatenate[n]])

            layer = decoder_block(layer, concatenate[n], filter_base*2**n, dropout=dropout,batchnorm=False,activation='linear')
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i))(layer)
            layer = conv_blocks(filter_base * 2 ** max(0,(n-1)), kernel, dropout=dropout,
                                batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, convs_per_depth))(layer)

        return layer
    return _func

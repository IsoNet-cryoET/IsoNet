from IsoNet.models.unet.blocks import conv_blocks, activation_my, decoder_block
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, AveragePooling3D,Conv2D,Add,Conv2DTranspose,Conv3D,Conv3DTranspose,Dropout,BatchNormalization,Activation,LeakyReLU
from tensorflow.keras.layers import Concatenate

# define a decoder block

def build_unet(filter_base=32,depth=3,convs_per_depth=3,
               kernel=(3,3,3),
               batch_norm=True,
               dropout=0.0,
               pool=None):
    resnet = False
    # pool = (2,2,2)
    def _func(inputs):
        concatenate = []
        layer = inputs
        #begin contracting path
        for n in range(depth):
            current_depth_start = layer
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base*2**n,kernel,dropout=dropout,
                                    batch_norm=batch_norm,activation = "LeakyReLU",
                                    name="down_level_%s_no_%s" % (n, i))(layer)
            # if use res_block strategy
            if resnet:  
                start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(current_depth_start)
                layer = Add()([start_conv,layer])
                layer = activation_my("LeakyReLU")(layer)
            # save the last layer of current depth
            concatenate.append(layer)
            # dimension reduction with pooling or stride 2 convolution
            if pool is not None:
                layer = MaxPooling3D(pool)(layer)
            else:
                layer = conv_blocks(filter_base*2**n,kernel,strides=(2,2,2),activation="LeakyReLU")(layer)
        # begin bottleneck path
        b = layer
        bottle_start = layer
        for i in range(convs_per_depth-2):
            b = conv_blocks(filter_base*2**depth,kernel,dropout=None,
                                    batch_norm=None,activation="LeakyReLU",
                                    name="bottleneck_no_%s" % (i))(b)
        layer = conv_blocks(filter_base*2**(depth-1),kernel,dropout=None,
                                    batch_norm=None,activation=None,
                                    name="bottleneck_no_%s" % (convs_per_depth))(b)
        if resnet:
            layer = Add()([bottle_start,layer])
            layer = activation_my("LeakyReLU")(layer)


        for n in reversed(range(depth)):
            if pool is not None:
                layer = Concatenate(axis=-1)([UpSampling3D(pool)(layer),concatenate[n]])
            else:
                layer = decoder_block(layer, concatenate[n], filter_base*2**n, dropout=False,batchnorm=False,activation="LeakyReLU")
            current_depth_start = layer
            for i in range(convs_per_depth):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i),activation ="LeakyReLU")(layer)
            if resnet:
                start_conv = Conv3D(filter_base*2**n,(1,1,1),
                            padding='same',kernel_initializer="he_uniform")(current_depth_start)
                layer = Add()([start_conv,layer])
                layer = activation_my("LeakyReLU")(layer)
        final = conv_blocks(1, (1,1,1), dropout=None,activation='linear',
                                    batch_norm=None,name="fullconv_out")(layer)
        return final
    return _func

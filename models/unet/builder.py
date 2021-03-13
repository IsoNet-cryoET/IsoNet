from IsoNet.models.unet.blocks import conv_blocks
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, MaxPooling3D, UpSampling3D, AveragePooling3D
from tensorflow.keras.layers import Concatenate

def build_unet(filter_base=32,depth=2,convs_per_depth=2,
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

            layer = Concatenate(axis=-1)([UpSampling(pool)(layer),concatenate[n]])
            for i in range(convs_per_depth-1):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i))(layer)
            layer = conv_blocks(filter_base * 2 ** max(0,(n-1)), kernel, dropout=dropout,
                                batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, convs_per_depth))(layer)

        return layer
    return _func

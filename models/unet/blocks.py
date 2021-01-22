from tensorflow.keras.layers import Dropout, Activation, BatchNormalization, Conv2D, Conv3D, LeakyReLU

def conv_blocks(n_filter, kernel,
                activation="relu",
                padding="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):
    def layer(last_layer):
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
        if dropout is not None and dropout>0:
            conv=Dropout(dropout)(conv)
        # conv=Activation(activation)(conv)
        conv = LeakyReLU(alpha=0.05)(conv)
        return conv
    return layer

from IsoNet.models.unet import builder,builder_fullconv,builder_fullconv_old,build_old_net
from tensorflow.keras.layers import Input,Add,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
def Unet(filter_base=32,
        depth=3,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=True,
        dropout=0.3,
        pool=None,residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004,
        test_shape=None):

    # model = builder.build_unet(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            pool)
    model = builder_fullconv.build_unet(filter_base,depth,convs_per_depth,
            kernel,
            batch_norm,
            dropout,
            pool)

    # model = build_old_net.unet_block(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            (2,2,2))
    # import os
    # import sys
    # cwd = os.getcwd()
    # sys.path.insert(0,cwd)
    # import train_settings 
    # model = builder_fullconv_old.build_unet(train_settings)
    
    #***** Construct complete model from unet output
    if test_shape is None:
        inputs = Input((None,None,None,1))
    elif type(test_shape) is int:
        inputs = Input((test_shape,test_shape,test_shape,1))
    unet_out = model(inputs) 
    if residual:
        outputs = Add()([unet_out, inputs])
    else:
        outputs = unet_out
    # outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=lr)
    if loss == 'mae' or loss == 'mse':
        metrics = ('mse', 'mae')

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

if __name__ == "__main__":
    keras_model = Unet(filter_base=64,
        depth=3,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=True,
        dropout=0.5,
        pool=(2,2,2),residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004)
    print(keras_model.summary())

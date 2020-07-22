from keras.layers import Dropout, Activation, BatchNormalization
from keras.layers import Add, Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, AveragePooling2D, AveragePooling3D, Conv3DTranspose
from keras.initializers import RandomNormal
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import Sequence
import keras.backend as K
import numpy as np
from keras.utils import multi_gpu_model
from keras import regularizers
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
        g = BatchNormalization()(g, training=True)
    if dropout is not None and dropout>0:
        g=Dropout(dropout)(g, training=True)
    g = Activation(activation)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, kernel=(3,3,3), strides=(2,2,2), dropout=0.5, batchnorm=True,activation='relu'):
    # weight initialization
    init = RandomNormal(stddev=0.02)
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
    g = Activation(activation)(g)
    return g

def encoder_layer(layer_in,depth,settings):
    # e = define_encoder_block(layer_in,settings.filter_base*2**n,strides=(2,2,2))(layer_in)
    e = layer_in
    for i in range(settings.nconvs_per):
        e = encoder_block(e, settings.filter_base*2**depth, strides=(1,1,1))
    return e   

def define_unet_generator(image_shape,settings):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    inp = Input(shape=image_shape)
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
    b = Activation('relu')(b)
    b = Conv3D(settings.filter_base*2**3, (3,3,3), strides=(1,1,1), padding='same', kernel_initializer=init)(b)
    # decoder model
    d3_ups = decoder_block(b, l3_out, settings.filter_base*2**3, dropout=None,batchnorm=False,activation='linear')
    d3_out = encoder_layer(d3_ups,3,settings)
    d2_ups = decoder_block(d3_out,l2_out,settings.filter_base*2**2,dropout=None,batchnorm=False,activation='linear')
    d2_out = encoder_layer(d2_ups,2,settings)
    d1_ups = decoder_block(d2_out,l1_out,settings.filter_base*2**1,dropout=None,batchnorm=False,activation='linear')
    d1_out = encoder_layer(d1_ups,1,settings)
    final = Conv3D(1,(1,1,1), strides=(1,1,1), padding='same', kernel_initializer=init)(d1_out)
    if  settings.residual == True:
        final = Add()([final,inp])
    out_image = Activation(settings.last_activation)(final)
    # define model
    model = Model(inp, out_image)
    return model

def train3D_seq(outFile, data_folder = 'data', epochs=40, steps_per_epoch=128,batch_size=32, n_gpus=1,loss='mae'):
    sys.path.insert(0,os.getcwd())
    if os.path.isfile('train_settings.py'):
        print('train_settings from cwd')
        import train_settings
    else:
        from mwr.models import train_settings 
    optimizer = Adam(train_settings.lr)
    metrics = train_settings.metrics
    # _metrics = [eval('loss_%s()' % m) for m in metrics]

    # inputs = Input((None, None,None, 1))
    model = define_unet_generator((None, None,None, 1),train_settings)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    print('***',train_settings.loss)
    model.compile(optimizer=optimizer, loss=train_settings.loss, metrics=train_settings.metrics)

    train_data, test_data = prepare_dataseq(data_folder, batch_size)
    print('**train data size**',len(train_data))
    
    callback_list = []
    check_point = ModelCheckpoint('results/modellast.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)
    
    history = model.fit_generator(generator=train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=callback_list)

    model.save_weights(outFile)
    return history


def train3D_continue(outFile, weights,data_folder = 'data', epochs=40, steps_per_epoch=128, batch_size=64,n_gpus=2,loss='mae'):

    metrics = ('mse', 'mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    optimizer = Adam(lr=0.0003)

    from keras.models import model_from_json

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()

    json_file.close()
    model = model_from_json(loaded_model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)

    model.load_weights(weights)
    print("Loaded model from disk")


    model.compile(optimizer=optimizer, loss=loss, metrics=_metrics)


    train_data, test_data = prepare_dataseq(data_folder, batch_size)

    callback_list = []
    check_point = ModelCheckpoint('results/modellast.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
    callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)

    history = model.fit_generator(generator=train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=callback_list)

    model.save(outFile)
    return history
    
def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)

def loss_mae(mean=True):
    R = _mean_or_not(mean)
    def mae(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.abs(y_pred[...,:n] - y_true))
    return mae


def loss_mse(mean=True):
    R = _mean_or_not(mean)
    def mse(y_true, y_pred):
        n = K.shape(y_true)[-1]
        return R(K.square(y_pred[...,:n] - y_true))
    return mse

class dataSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.x))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]

        rx = np.array([mrcfile.open(self.x[j]).data[:,:,:,np.newaxis] for j in idx])
        ry = np.array([mrcfile.open(self.y[j]).data[:,:,:,np.newaxis] for j in idx])
        return rx,ry

def prepare_dataseq(data_folder, batch_size):
    import os
    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    path_all = []
    for d in dirs_tomake:
        p = '{}/{}/'.format(data_folder, d)
        path_all.append(sorted([p+f for f in os.listdir(p)]))
    train_data = dataSequence(path_all[0], path_all[1], batch_size)
    test_data = dataSequence(path_all[2], path_all[3], batch_size)
    return train_data, test_data


'''
date 4/11/2018
heng zhang
'''
from keras.layers import Dropout, Activation, BatchNormalization
from keras.layers import Add, Input, Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, UpSampling3D, AveragePooling2D, AveragePooling3D
from keras.layers.merge import Concatenate
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import Sequence
import keras.backend as K
import numpy as np
from keras.utils import multi_gpu_model
from keras import regularizers

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
                        #kernel_regularizer=regularizers.l2(0.00005),
                        kernel_initializer=init,**kwargs)(last_layer)
        elif len(kernel)==3:
            conv=Conv3D(n_filter,kernel,
                        padding=padding,
                        #kernel_regularizer=regularizers.l2(0.0005),
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
        #MaxPooling=MaxPooling3D
        MaxPooling=AveragePooling3D
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
            print('depth is : %d' % n)

            layer = Concatenate(axis=-1)([UpSampling(pool)(layer),concatenate[n]])
            for i in range(convs_per_depth-1):
                layer = conv_blocks(filter_base * 2 ** n, kernel, dropout=dropout,
                                    batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, i))(layer)
            layer = conv_blocks(filter_base * 2 ** max(0,(n-1)), kernel, dropout=dropout,
                                batch_norm=batch_norm,name="up_level_%s_no_%s" % (n, convs_per_depth))(layer)

        return layer
    return _func
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





class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]


def train(X,Y, val_data, outFile, epochs=20,steps_per_epoch=64,batch_size=128,n_gpus=6):
    filter_base=32
    depth = 2
    convs_per_depth = 2
    kernel = (5, 5)
    batch_norm = False
    dropout = 0.0
    pool = (2, 2)
    last_activation='linear'
    optimizer = Adam(lr=0.0004)
    metrics = ('mse','mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    residual = True

    print(kernel)
    print(len(kernel))
    inputs=Input((None,None,1))
    Unet = unet_block(filter_base=filter_base, depth=depth,convs_per_depth=convs_per_depth,
                      kernel=kernel,
                      batch_norm=batch_norm,dropout=dropout,
                      pool=pool)(inputs)
    if len(kernel)==2:
        final = Conv2D(1,(1,1),activation='linear')(Unet)
    elif len(kernel)==3:
        final = Conv3D(1,(1,1,1),activation='linear')(Unet)
    if residual:
        final = Add()([final, inputs])


    outputs = Activation(activation=last_activation)(final)
    model = Model(inputs=inputs,outputs=outputs)
    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    # model.summary()
    # return model
    model.compile(optimizer=optimizer,loss='mae',metrics=_metrics)


    train_data = DataWrapper(X,Y,batch_size)
    history = model.fit_generator(generator=train_data, validation_data=val_data,
                                  epochs=epochs,steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=[TensorBoard(log_dir='./Graph', histogram_freq= 0, write_graph=True, write_images=True)])

    model.save(outFile)
    return history


def train3D(outFile, data_folder = 'data', epochs=40, steps_per_epoch=128,batch_size=32, dropout = 0.3,filter_base=32, convs_per_depth = 3,kernel = (3,3,3), pool = (2,2,2), batch_norm = False, depth = 3, n_gpus=2):
    filter_base = filter_base
    depth = depth
    convs_per_depth = convs_per_depth
    kernel = kernel
    batch_norm = batch_norm
    dropout = dropout
    pool = pool
    last_activation = 'linear'
    optimizer = Adam(lr=0.0004)
    metrics = ('mse', 'mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    residual = True

    inputs = Input((None, None,None, 1))
    Unet = unet_block(filter_base=filter_base, depth=depth, convs_per_depth=convs_per_depth,
                      kernel=kernel,
                      batch_norm=batch_norm, dropout=dropout,
                      pool=pool)(inputs)
    if len(kernel) == 2:
        outputs = Conv2D(1, (1, 1), activation='linear')(Unet)
    elif len(kernel) == 3:
        outputs = Conv3D(1, (1, 1, 1), activation='linear')(Unet)
    if residual:
        outputs = Add()([outputs, inputs])

    outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    #model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #if mrc_list is not None:
    #    model.compile(optimizer=optimizer, loss=loss_custom(model,read_data_mrc(mrc_list)), metrics=_metrics)
    #else:
    model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)

    train_data, test_data = prepare_dataseq(data_folder, batch_size)

    callback_list = []
    check_point = ModelCheckpoint('results/weights{epoch:08d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5)
    callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)
    
    history = model.fit_generator(generator=train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=callback_list)

    model.save_weights(outFile)
    return history






import numpy as np
from tifffile import imread
# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
import mrcfile
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


def train3D_seq(outFile, data_folder = 'data', epochs=40, steps_per_epoch=128,batch_size=32, dropout = 0.3,filter_base=32, convs_per_depth = 3,kernel = (3,3,3), pool = (2,2,2), batch_norm = False, depth = 3, n_gpus=2):
    filter_base = filter_base
    depth = depth
    convs_per_depth = convs_per_depth
    kernel = kernel
    batch_norm = batch_norm
    dropout = dropout
    pool = pool
    last_activation = 'linear'
    optimizer = Adam(lr=0.0004)
    metrics = ('mse', 'mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    residual = True

    inputs = Input((None, None,None, 1))
    Unet = unet_block(filter_base=filter_base, depth=depth, convs_per_depth=convs_per_depth,
                      kernel=kernel,
                      batch_norm=batch_norm, dropout=dropout,
                      pool=pool)(inputs)
    if len(kernel) == 2:
        outputs = Conv2D(1, (1, 1), activation='linear')(Unet)
    elif len(kernel) == 3:
        outputs = Conv3D(1, (1, 1, 1), activation='linear')(Unet)
    if residual:
        outputs = Add()([outputs, inputs])

    outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    #model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #if mrc_list is not None:
    #    model.compile(optimizer=optimizer, loss=loss_custom(model,read_data_mrc(mrc_list)), metrics=_metrics)
    #else:
    model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)

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

def train3D_continue(outFile, weights,data_folder = 'data', epochs=40, steps_per_epoch=128, batch_size=64,n_gpus=2 ):

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


    model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)


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


if __name__ == '__main__':
    data = np.load('/home/heng/small_train_data.npz')
    x=data['x']
    y=data['y']
    x_val=data['x_val']
    y_val=data['y_val']
    train(x,y,(x_val,y_val))
'''
# ------------ save the template model rather than the gpu_mode ----------------
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# -------------- load the saved model --------------
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer='adadelta',
                     metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''





































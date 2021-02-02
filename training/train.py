from tensorflow.keras.layers import Activation, Add, Input, Conv2D, Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import multi_gpu_model
from mwr.training.data_sequence import prepare_dataseq
from mwr.models.unet.model import Unet
import tensorflow as tf
from mwr.losses.losses import loss_mae,loss_mse

from tensorflow.keras.models import model_from_json,load_model, clone_model
import os
import logging


def train3D_seq(outFile,
                data_dir = 'data',
                result_folder='results',
                epochs=40,
                steps_per_epoch=128,
                batch_size=32,
                dropout = 0.3,
                lr=0.0004,
                filter_base=32,
                convs_per_depth = 3,
                kernel = (3,3,3),
                pool = (2,2,2),
                batch_norm = False,
                depth = 3,
                n_gpus=2,
                last_activation = 'linear',
                residual = True,
                loss = 'mae'):

    # optimizer = Adam(lr=lr)
    # if loss == 'mae' or loss == 'mse':
    #     metrics = ('mse', 'mae')
    #     _metrics = [eval('loss_%s()' % m) for m in metrics]
    # elif loss == 'binary_crossentropy':
    #     _metrics = ['accuracy']


    strategy = tf.distribute.MirroredStrategy()
    if n_gpus > 1:
        with strategy.scope():
            # model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)    
            model = Unet(filter_base=filter_base, 
                depth=depth, 
                convs_per_depth=convs_per_depth,
                kernel=kernel,
                batch_norm=batch_norm, 
                dropout=dropout,
                pool=pool,
                residual = residual,
                last_activation = last_activation,
                loss = loss,
                lr = lr)
            # model.compile(optimizer=optimizer, loss=loss, metrics=_metrics)
    else:
        model = Unet(filter_base=filter_base, 
            depth=depth, 
            convs_per_depth=convs_per_depth,
            kernel=kernel,
            batch_norm=batch_norm, 
            dropout=dropout,
            pool=pool,
            residual = residual,
            last_activation = last_activation,
            loss = loss,
            lr = lr)
        # model.compile(optimizer=optimizer, loss=loss, metrics=_metrics)
    print(model.summary())
    train_data, test_data = prepare_dataseq(data_dir, batch_size)
    print('**train data size**',len(train_data))
    callback_list = []
    # check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder),
    #                             monitor='val_loss',
    #                             verbose=0,
    #                             save_best_only=False,
    #                             save_weights_only=False,
    #                             mode='auto',
    #                             period=1)
    # callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)
    history = model.fit_generator(generator=train_data,
                                validation_data=test_data,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                verbose=1)
                                # callbacks=callback_list)

    # if n_gpus>1:
    #     model_from_multimodel = model.get_layer('model_1')   
    #     model_from_multimodel.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #     model_from_multimodel.save(outFile)
    # else:
    model.save(outFile)

    return history

def train3D_continue(outFile,
                    model_file,
                    data_dir = 'data',
                    result_folder='results',
                    epochs=40,
                    lr=0.0004,
                    steps_per_epoch=128,
                    batch_size=64,
                    n_gpus=2):
    
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
    logging.debug('The tf message level {}'.format(os.environ['TF_CPP_MIN_LOG_LEVEL']))
    # metrics = ('mse', 'mae')
    # _metrics = [eval('loss_%s()' % m) for m in metrics]
    # optimizer = Adam(lr=lr)
    
    # model = load_model( model_file) # weight is a model
    strategy = tf.distribute.MirroredStrategy()
    if n_gpus > 1:
        with strategy.scope():
            model = load_model( model_file)
        # model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    else:
        model = load_model( model_file)

    # model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    # model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    logging.info("Loaded model from disk")


    train_data, test_data = prepare_dataseq(data_dir, batch_size)

    callback_list = []
    # check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder),
    #                             monitor='val_loss',
    #                             verbose=0,
    #                             save_best_only=False,
    #                             save_weights_only=False,
    #                             mode='auto',
    #                             period=1)
    # callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)
    logging.info("begin fitting")
    history = model.fit(train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=1)
                                #   callbacks=callback_list)
    # if n_gpus>1:
    #     model_from_multimodel = model.get_layer('model_1')   
    #     model_from_multimodel.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #     model_from_multimodel.save(outFile)
    # else:
    model.save(outFile)
    return history

def prepare_first_model(settings):
    model = Unet(filter_base=settings.filter_base, 
            depth=settings.unet_depth, 
            convs_per_depth=settings.convs_per_depth,
            kernel=settings.kernel,
            batch_norm=settings.batch_normalization, 
            dropout=settings.drop_out,
            pool=settings.pool,
            residual = True,
            last_activation = 'linear',
            loss = 'mae',
            lr = settings.lr)
    init_model_name = settings.result_dir+'/model_init.h5'
    model.save(init_model_name)
    settings.init_model = init_model_name
    return settings

def train_data(settings):
    history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count),
                                        settings.init_model,
                                        data_dir = settings.data_dir,
                                        result_folder = settings.result_dir,
                                        epochs=settings.epochs,
                                        steps_per_epoch=settings.steps_per_epoch,
                                        batch_size=settings.batch_size,
                                        lr = settings.lr,
                                        n_gpus=settings.ngpus)

    # if settings.iter_count == 0 and settings.pretrained_model is None :
    #     history = train3D_seq('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                 data_dir = settings.data_dir,
    #                                 result_folder = settings.result_dir,
    #                                 epochs = settings.epochs,
    #                                 steps_per_epoch = settings.steps_per_epoch,
    #                                 batch_size = settings.batch_size,
    #                                 lr = settings.lr,
    #                                 dropout = settings.drop_out,
    #                                 filter_base = settings.filter_base,
    #                                 depth=settings.unet_depth,
    #                                 convs_per_depth = settings.convs_per_depth,
    #                                 batch_norm = settings.batch_normalization,
    #                                 kernel = settings.kernel,
    #                                 n_gpus = settings.ngpus)
    # elif settings.iter_count == 0 and settings.pretrained_model is not None:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     settings.pretrained_model,
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     lr = settings.lr,
    #                                     n_gpus=settings.ngpus)

    # else:
    #     history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1),
    #                                     '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count),
    #                                     data_dir = settings.data_dir,
    #                                     result_folder = settings.result_dir,
    #                                     epochs=settings.epochs,
    #                                     steps_per_epoch=settings.steps_per_epoch,
    #                                     batch_size=settings.batch_size,
    #                                     n_gpus=settings.ngpus)

    return history
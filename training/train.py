from keras.layers import Activation, Add, Input, Conv2D, Conv3D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import Sequence
from keras.utils import multi_gpu_model
from training.data_sequence import prepare_dataseq
from models.unet.model import Unet 
from losses.losses import loss_mae,loss_mse

from keras.models import model_from_json

import logging

def train3D_seq_old(outFile, 
	data_folder,
	epochs=40, 
	steps_per_epoch=128,
	batch_size=64,
	dropout = 0.3,
	lr = 0.001,
	filter_base=32,
	convs_per_depth = 3,
	kernel = (3,3,3), 
	pool = (2,2,2), 
	batch_norm = False, 
	depth = 3, 
	n_gpus=1):

    last_activation = 'softmax'
    optimizer = Adam(lr)
    #metrics = ('mse', 'categorical_crossentropy')
    #_metrics = [eval('loss_%s()' % m) for m in metrics]
    residual = False

    inputs = Input((None, None,None, 1))

    unet = Unet(filter_base=filter_base, 
    	depth=depth, 
    	convs_per_depth=convs_per_depth,
        kernel=kernel,
        batch_norm=batch_norm, 
        dropout=dropout,
        pool=pool)(inputs)
    if len(kernel) == 2:
        outputs = Conv2D(1, (1, 1), activation='linear')(unet)
    elif len(kernel) == 3:
        outputs = Conv3D(21, (1, 1, 1))(unet)
    if residual:
        final = Add()([outputs, inputs])

    outputs = Activation(activation=last_activation)(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    model_json = model.to_json()

    with open("{}model.json".format(settings.ab_result_folder), "w") as json_file:
        json_file.write(model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    #model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #if mrc_list is not None:
    #    model.compile(optimizer=optimizer, loss=loss_custom(model,read_data_mrc(mrc_list)), metrics=_metrics)
    #else:
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')#, metrics=_metrics)

    train_data, test_data = prepare_dataseq(data_folder, batch_size)

    callback_list = []
    check_point = ModelCheckpoint('{}modellast.h5'.format(settings.ab_result_folder), 
    	monitor='val_loss', 
    	verbose=0, 
    	save_best_only=False, 
    	save_weights_only=True, 
    	mode='auto', period=1)
    callback_list.append(check_point)

    tensor_board = TensorBoard(log_dir='{}Graph'.format(settings.ab_result_folder), 
    	histogram_freq=0, 
    	write_graph=True, 
    	write_images=True)
    callback_list.append(tensor_board)

    history = model.fit_generator(generator=train_data, 
    	validation_data=test_data,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        callbacks=callback_list)

    model.save_weights(outFile)
    return history

def train3D_seq(outFile, 
                data_folder = 'data', 
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
                n_gpus=2):

    last_activation = 'linear'
    optimizer = Adam(lr)
    metrics = ('mse', 'mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    residual = True

    inputs = Input((None, None,None, 1))
    unet = Unet(filter_base=filter_base, 
        depth=depth, 
        convs_per_depth=convs_per_depth,
        kernel=kernel,
        batch_norm=batch_norm, 
        dropout=dropout,
        pool=pool)(inputs)
    #Unet = unet_block(filter_base=filter_base, depth=depth, convs_per_depth=convs_per_depth,
    #                  kernel=kernel,
    #                  batch_norm=batch_norm, dropout=dropout,
    #                  pool=pool)(inputs)
    if len(kernel) == 2:
        outputs = Conv2D(1, (1, 1), activation='linear')(unet)
    elif len(kernel) == 3:
        outputs = Conv3D(1, (1, 1, 1), activation='linear')(unet)
    if residual:
        outputs = Add()([outputs, inputs])

    outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    model_json = model.to_json()
    with open("{}/model.json".format(result_folder), "w") as json_file:
        json_file.write(model_json)

    if n_gpus > 1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)
    #model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)
    #if mrc_list is not None:
    #    model.compile(optimizer=optimizer, loss=loss_custom(model,read_data_mrc(mrc_list)), metrics=_metrics)
    #else:
    model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)

    train_data, test_data = prepare_dataseq(data_folder, batch_size)
    logging.info('**train data size**',len(train_data))
    
    callback_list = []
    check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder), 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=True, 
                                mode='auto', 
                                period=1)
    callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)
    
    history = model.fit_generator(generator=train_data, 
                                validation_data=test_data,
                                epochs=epochs, 
                                steps_per_epoch=steps_per_epoch,
                                verbose=1,
                                callbacks=callback_list)

    model.save_weights(outFile)
    return history

def train3D_continue(outFile, 
                    weights,
                    data_folder = 'data',
                    result_folder='results', 
                    epochs=40, 
                    steps_per_epoch=128, 
                    batch_size=64,
                    n_gpus=2):

    metrics = ('mse', 'mae')
    _metrics = [eval('loss_%s()' % m) for m in metrics]
    optimizer = Adam(lr=0.0003)

    json_file = open('{}/model.json'.format(result_folder), 'r')
    loaded_model_json = json_file.read()

    json_file.close()
    model = model_from_json(loaded_model_json)

    if n_gpus >1:
        model = multi_gpu_model(model, gpus=n_gpus, cpu_merge=True, cpu_relocation=False)

    model.load_weights(weights)
    logging.info("Loaded model from disk")

    model.compile(optimizer=optimizer, loss='mae', metrics=_metrics)


    train_data, test_data = prepare_dataseq(data_folder, batch_size)

    callback_list = []
    check_point = ModelCheckpoint('{}/modellast.h5'.format(result_folder), 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=True, 
                                mode='auto', 
                                period=1)
    callback_list.append(check_point)
    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    callback_list.append(tensor_board)

    history = model.fit_generator(generator=train_data, validation_data=test_data,
                                  epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  verbose=1,
                                  callbacks=callback_list)

    model.save(outFile)
    return history


def train_data(settings):

    if settings.iter_count == 0 or not settings.reload_weight:
        history = train3D_seq('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1), 
                                    data_folder = settings.data_folder,
                                    result_folder = settings.result_dir, 
                                    epochs = settings.epochs, 
                                    steps_per_epoch = settings.steps_per_epoch,  
                                    batch_size = settings.batch_size, 
                                    dropout = settings.drop_out, 
                                    depth=settings.unet_depth,
                                    convs_per_depth = settings.convs_per_depth,
                                    batch_norm = settings.batch_normalization, 
                                    kernel = settings.kernel, 
                                    n_gpus = settings.ngpus)
    else:
        history = train3D_continue('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1), 
                                        '{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count), 
                                        data_folder = settings.data_folder, 
                                        result_folder = settings.result_dir, 
                                        epochs=settings.epochs, 
                                        steps_per_epoch=settings.steps_per_epoch, 
                                        batch_size=settings.batch_size, 
                                        n_gpus=settings.ngpus)

    return history

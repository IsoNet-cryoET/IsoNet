from keras.models import model_from_json
from keras.utils import multi_gpu_model
import mrcfile
from preprocessing.img_processing import normalize
import numpy as np
import logging

def predict(settings):


    json_file = open('{}/model.json'.format(settings.result_dir), 'r')
    loaded_model_json = json_file.read()

    json_file.close()
    model = model_from_json(loaded_model_json)

    if settings.ngpus >1:
        model = multi_gpu_model(model, gpus=settings.ngpus, cpu_merge=True, cpu_relocation=False)
    model.load_weights('{}/model_iter{:0>2d}.h5'.format(settings.result_dir,settings.iter_count+1))

    N = settings.predict_batch_size * settings.ngpus

    num_batches = len(settings.mrc_list)
    if num_batches%N == 0:
        append_number = 0
    else:
        append_number = N - num_batches%N
    data = []

    for i,mrc in enumerate(settings.mrc_list + settings.mrc_list[:append_number]):
        root_name = mrc.split('/')[-1].split('.')[0]
        if i < len(settings.mrc_list):
            print('predicting:{}'.format(root_name))

        with mrcfile.open('{}/{}_iter00.mrc'.format(settings.result_dir,root_name)) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
        real_data=normalize(real_data, percentile = settings.normalize_percentile)

        cube_size = real_data.shape[0]
        pad_size1 = (settings.predict_cropsize - cube_size)//2
        pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
        padi = (pad_size1,pad_size2)
        real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

        if (i+1)%N != 0:
            data.append(real_data)
        else:
            data.append(real_data)
            data = np.array(data)
            predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=1)
            predicted = predicted.reshape(predicted.shape[0:-1])
            logging.info(predicted)
            for j,outData in enumerate(predicted):
                count = i + j - N + 1
                if count < len(settings.mrc_list):
                    m_name = settings.mrc_list[count]

                    root_name = m_name.split('/')[-1].split('.')[0]
                    end_size = pad_size1+cube_size
                    outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                    outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                    with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(settings.result_dir,root_name,settings.iter_count+1), overwrite=True) as output_mrc:
                        output_mrc.set_data(-outData1)
            data = []

    if settings.tomogram2_list is not None:
        data = []

        for i,mrc in enumerate(settings.mrc_list + settings.mrc_list[:append_number]):
            root_name = mrc.split('/')[-1].split('.')[0]
            if i < len(settings.mrc_list):
                print('predicting:{}'.format(root_name))

            with mrcfile.open('{}/{}_iter00.mrc2'.format(settings.result_dir,root_name)) as mrcData:
                real_data = mrcData.data.astype(np.float32)*-1
            real_data=normalize(real_data, percentile = settings.normalize_percentile)

            cube_size = real_data.shape[0]
            pad_size1 = (settings.predict_cropsize - cube_size)//2
            pad_size2 = pad_size1+1 if (settings.predict_cropsize - cube_size)%2 !=0 else  pad_size1
            padi = (pad_size1,pad_size2)
            real_data = np.pad(real_data, (padi,padi,padi), 'symmetric')

            if (i+1)%N != 0:
                data.append(real_data)
            else:
                data.append(real_data)
                data = np.array(data)
                print('***',data.shape)
                predicted=model.predict(data[:,:,:,:,np.newaxis], batch_size= settings.predict_batch_size,verbose=1)
                predicted = predicted.reshape(predicted.shape[0:-1])

                for j,outData in enumerate(predicted):
                    count = i + j - N + 1
                    if count < len(settings.mrc_list):
                        m_name = settings.mrc_list[count]
                        root_name = m_name.split('/')[-1].split('.')[0]
                        end_size = pad_size1+cube_size
                        outData1 = outData[pad_size1:end_size, pad_size1:end_size, pad_size1:end_size]
                        outData1 = normalize(outData1, percentile = settings.normalize_percentile)
                        with mrcfile.new('{}/{}_iter{:0>2d}.mrc2'.format(settings.result_dir,root_name,settings.iter_count+1), overwrite=True) as output_mrc:
                            output_mrc.set_data(-outData1)
                data = []        

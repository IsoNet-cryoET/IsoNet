import json,sys
global refine_param, predict_param, extract_param, param_to_check, param_to_set_attr
refine_param = [ 'normalize_percentile', 'batch_normalization', 'filter_base', 'unet_depth', 'pool', 'kernel', 'convs_per_depth', 'drop_out', 'noise_mode', 'noise_pause', 'noise_start_iter', 'noise_level', 'steps_per_epoch', 'batch_size', 'epochs', 'continue_train', 'preprocessing_ncpus', 'result_dir', 'continue_iter', 'log_level', 'pretrained_model', 'data_folder', 'iterations', 'gpuID', 'subtomo_star']
predict_param = ['tomo_idx', 'Ntile', 'log_level', 'normalize_percentile', 'batch_size', 'use_deconv_tomo', 'crop_size', 'cube_size', 'gpuID', 'output_dir', 'model', 'star_file']
extract_param = ['log_level', 'cube_size', 'subtomo_star', 'subtomo_folder', 'use_deconv_tomo', 'star_file']
param_to_check = refine_param + predict_param + extract_param + ['self','run']
param_to_set_attr = refine_param + predict_param + extract_param
class Arg:
    def __init__(self,dictionary):
        import logging
        # Do not set global basic logging 
        #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        #datefmt='%Y-%m-%d:%H:%M:%S')
        #logger = logging.getLogger('IsoNet.util.dict2attr')
        for k, v in dictionary.items():
            if k not in param_to_check:
                logging.warning("{} not recognized!".format(k))
                sys.exit(0)
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            if k in param_to_set_attr:
                setattr(self, k, v)
            # print(k,v)
            # param_list.append(k)
def save_args_json(args,file_name):
    encoded = json.dumps(args.__dict__, indent=4, sort_keys=True)
    with open(file_name,'w') as f:
        f.write(encoded)

def load_args_from_json(file_name):
    with open(file_name,'r') as f:
        contents = f.read()
    encoded = json.loads(contents)
    return Arg(encoded)

def check_args(args):
    train_params = ['self','train','normalize_percentile', 'batch_normalization', 'filter_base', 'unet_depth', 'kernel', 'convs_per_depth', 'drop_out', 'steps_per_epoch', 'batch_size', 'epochs', 'preprocessing_ncpus', 'ncube','filter_base', 'crop_size', 'cube_size', 'noise_pause', 'noise_start_iter', 'noise_level', 'continue_iter', 'continue_training', 'log_level', 'pretrained_model', 'data_dir', 'subtomo_dir', 'datas_are_subtomos', 'iterations', 'noise_dir', 'mask_dir', 'gpuID', 'input_dir']
    predict_params = ['self', 'predict','norm', 'batch_size', 'crop_size', 'cube_size', 'gpuID', 'model', 'output_file', 'mrc_file']
    mask_param = ['tomo_path','mask_name,side','percentile','threshold']
    noise_param = ['output_folder', 'number_volume', 'cubesize', 'minangle', 'maxangle','anglestep', 'start','ncpus', 'mode']
    param_list = train_params + predict_params + mask_param + noise_param
    for i in args[2:]:
        if i not in param_list:
            pass
            # logging.warning("{} not recognized!".format(i))

def idx2list(tomo_idx):
    if tomo_idx is not None:
            if type(tomo_idx) is tuple:
                tomo_idx = list(map(str,tomo_idx))
            elif type(tomo_idx) is int:
                tomo_idx = [str(tomo_idx)]
            else:
                # tomo_idx = tomo_idx.split(',')
                txt=str(tomo_idx)
                txt=txt.replace(',',' ').split()
                tomo_idx=[]
                for everything in txt:
                    if everything.find("-")!=-1:
                        everything=everything.split("-")
                        for e in range(int(everything[0]),int(everything[1])+1):
                            tomo_idx.append(str(e))
                    else:
                        tomo_idx.append(str(everything))
    return tomo_idx

def txtval(txt):
    txt=str(txt)
    txt=txt.replace(',',' ').split()
    idx=[]
    for everything in txt:
        if everything.find("-")!=-1:
            everything=everything.split("-")
            for e in range(int(everything[0]),int(everything[1])+1):
                idx.append(e)
        else:
            idx.append(int(everything))
    return idx
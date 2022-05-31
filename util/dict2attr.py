import json,sys
import logging
global logger 
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
global refine_param, predict_param, extract_param, param_to_check, param_to_set_attr
refine_param = [ 'normalize_percentile', 'batch_normalization', 'filter_base', 'unet_depth', 'pool', 'kernel', 'convs_per_depth', 'drop_out','noise_dir', 
                'noise_mode', 'noise_pause', 'noise_start_iter','learning_rate', 'noise_level', 'steps_per_epoch', 'batch_size', 'epochs', 'continue_from', 
                'preprocessing_ncpus', 'result_dir', 'continue_iter', 'log_level', 'pretrained_model', 'data_dir', 'iterations', 'gpuID', 'subtomo_star','cmd',
                'select_subtomo_number','remove_intermediate']
predict_param = ['tomo_idx', 'Ntile', 'log_level', 'normalize_percentile', 'batch_size', 'use_deconv_tomo', 'crop_size', 'cube_size', 'gpuID', 'output_dir', 'model', 'star_file']
extract_param = ['log_level', 'cube_size', 'subtomo_star', 'subtomo_folder', 'use_deconv_tomo', 'star_file','tomo_idx','crop_size']
deconv_param = ['star_file', 'deconv_folder','chunk_size', 'snrfalloff', 'deconvstrength', 'highpassnyquist', 'tile', 'overlap_rate', 'ncpu', 'tomo_idx']
make_mask_param = ['star_file', 'mask_folder', 'patch_size', 'density_percentage', 'std_percentage', 'use_deconv_tomo', 'z_crop', 'tomo_idx', 'mask_boundary']
prepare_star_param = ['number_subtomos', 'defocus', 'pixel_size', 'output_star', 'folder_name']
prepare_subtomo_star_param = ['folder_name', 'output_star', 'pixel_size', 'cube_size']
param_to_check = refine_param + predict_param + extract_param + ['self','run']
param_to_set_attr = refine_param + predict_param + extract_param + ['iter_count','crop_size','cube_size','predict_cropsize','noise_dir','lr','ngpus','predict_batch_size','losses']
class Arg:
    def __init__(self,dictionary,from_cmd=True):
        for k, v in dictionary.items():
            if k not in param_to_check and from_cmd is True:
                logger.error("{} not recognized!".format(k))
                sys.exit(0)
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            if k == 'noise_start_iter' and type(v) is int:
                v = tuple([v])
            if k == 'noise_level' and type(v) in [int,float]:
                v = tuple([v])
            if k in param_to_set_attr:
                setattr(self, k, v)
         
def save_args_json(args,file_name):
    filtered_dict = Arg(args.__dict__,from_cmd=False)
    encoded = json.dumps(filtered_dict.__dict__, indent=4, sort_keys=True)
    with open(file_name,'w') as f:
        f.write(encoded)

def load_args_from_json(file_name):
    with open(file_name,'r') as f:
        contents = f.read()
    encoded = json.loads(contents)
    return Arg(encoded,from_cmd=False)

def check_parse(args_list):
    if args_list[0] in ['refine','predict','extract','deconv','make_mask','prepare_star','extract','prepare_subtomo_star','check','gui']:
        if args_list[0] in ['refine','predict','extract','deconv','make_mask','prepare_star','extract','prepare_subtomo_star']:
            check_list = eval(args_list[0]+'_param') + ['help']
        else:
            check_list = None
    else:
        check_list = None
        # logging.error(" '{}' is NOT a IsoNet function!".format(args_list[0]))
        # sys.exit(0)
    # check_list not None means need to check the parameters.
    if check_list is not None:
        for arg in args_list:
            if type(arg) is str and arg[0:2]=='--':
                if arg[2:] not in check_list:
                    logger.error(" '{}' not recognized!".format(arg[2:]))
                    sys.exit(0)


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

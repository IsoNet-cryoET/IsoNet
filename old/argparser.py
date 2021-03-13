import os
import argparse

def toTuple(s):
    try:
        x, y, z = s.split(',')
        return (int(x),int(y),int(z))
    except:
        raise argparse.ArgumentTypeError("kernel size must be x,y,z")





parser = argparse.ArgumentParser(
    description="Train MWR model", add_help=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--gpuID",
                    default='0',
                    help="The gpuID to used during the training. e.g 0,1,2,3.")

parser.add_argument("--datas_are_subtomos",
                    action="store_true",
                    default=False,
                    help="Is your training data subtomograms?")

parser.add_argument("input_dir",
                    default="input",
                    help="The folder where the input tomogram at.")

parser.add_argument("--subtomo_dir",
                    default="subtomograms",
                    help="The folder where the input subtomograms at.")

parser.add_argument("--result_dir",
                    default="results",
                    help="The folder where the input result model stored.")

parser.add_argument("--iterations",
                    type=int,
                    default=50,
                    help="Number of training iterations.")

parser.add_argument("--continue_previous_training",
                    action="store_true",
                    default=False,
                    help="Continus previous training? When continue, the architecture of network can not be changed.")

parser.add_argument("--continue_iter",
                    type=int,
                    default=0,
                    help="Which iteration you want to start from?")

parser.add_argument("--continue_from", 
                    choices=["preprocessing", "training" or "predicting"],
                    default="training",
                    help="Continue from either 'preprocessing', 'training' or 'predicting'.")

parser.add_argument("--log_level", 
                    choices=["debug" or "info"],
                    default="info",
                    help="logging level")

parser.add_argument("--reload_weight",
                    action="store_true",
                    default=True,
                    help="Reload weight from previous iterations. Keep this True.")

parser.add_argument("--preprocessing_ncpus",
                    type=int,
                    default=1,
                    help="Number of cpu for preprocessing.")

parser.add_argument("--data_folder",
                    default='data',
                    help="Temperary folder to save the generated data used for training.")

parser.add_argument("--mask",
                    default=None,
                    help="TODO: calculate mask to exclude empty area? Please set None for now.")

parser.add_argument("--cube_sidelen",
                    type=int,
                    default=80,
                    help="Size of training cubes, this size should be divisible by 2^unet_depth.")

parser.add_argument("--cropsize",
                    type=int,
                    default=120,
                    help="Size of cubes to impose missing wedge. Should be same or larger than size of cubes.")

parser.add_argument("--ncube",
                    type=int,
                    default=5,
                    help="Number of cubes generated for each (sub)tomos. Because each (sub)tomo rotates 16 times, the actual number of cubes for trainings should be ncube*16.")

parser.add_argument("--epochs",
                    type=int,
                    default=10,
                    help="Number of epoch for each iteraction.")

parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="Size of the minibatch.")

parser.add_argument("--steps_per_epoch",
                    type=int,
                    default=100,
                    help="Step per epoch. A good estimation of this value is (sub)tomos * ncube * 16 / batch_size *0.9.")

parser.add_argument("--noise_folder",
                    default=None,
                    help="Add noise during training, Set None to disable noise reduction.")

parser.add_argument("--noise_level",
                    type=float,
                    default=0.04,
                    help="Level of noise STD(added noise)/STD(data).")

parser.add_argument("--noise_start_iter",
                    type=int,
                    default=20,
                    help="Iteration to add trainning noise.")

parser.add_argument("--noise_pause",
                    type=int,
                    default=5,
                    help="Iters trainning noise remain at one level.")

parser.add_argument("--drop_out",
                    type=float,
                    default=0.5,
                    help="Drop out rate to reduce overfitting.")

parser.add_argument("--convs_per_depth",
                    type=int,
                    default=3,
                    help="Number of convolution layer for each depth.")

parser.add_argument('--kernel', 
                    help="Kernel for convolution layer. e.g 3,3,3", 
                    dest="kernel", 
                    default=(3,3,3),
                    type=toTuple)

parser.add_argument("--unet_depth",
                    type=int,
                    default=4,
                    help="Number of convolution layer for each depth.")

parser.add_argument("--batch_normalization",
                    action="store_true",
                    default=False,
                    help="Sometimes batch normalization may induce artifacts for extreme pixels in the first several iterations. Those could be restored in further iterations.")

parser.add_argument("--normalize_percentile",
                    action="store_true",
                    default=True,
                    help="Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.")

parser.add_argument("--predict_cropsize",
                    type=int,
                    default=128,
                    help="Predict cubesize.")

parser.add_argument("--predict_batch_size",
                    type=int,
                    default=8,
                    help="Predict batch size.")

parser.add_argument("--mrc_list",
                    default=None,
                    help="list of mrc cube files")

parser.add_argument("--mrc2_list",
                    default=None,
                    help="list of mrc2 cube files")

args = parser.parse_args()

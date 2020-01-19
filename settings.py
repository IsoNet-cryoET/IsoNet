
# computation settings
gpuID = "4,5,6,7"

#if your dataset is composes of full tomogram(s) set this value False
datas_are_subtomos = False
tomogram_list = ['pp366-bin2-5i.rec']
subtomo_dir = 'subtomos'

#number of iterations
iterations = 20

#continus previous training? When continue, the architecture of network can not be changed.
continue_previous_training =  False
continue_iter = 0

#continue from either "preprocessing", "training" or "predicting"
continue_from = "training"

#Reload weight from previous iterations. Keep this True
reload_weight = True

#-----------------------------------------------------------------------------------------
#preprocessing settings

#number of cpu for preprocessing
preprocessing_ncpus = 28

#Temperary folder to save the generated data used for training
data_folder = "/dev/shm/lytao/tmpdata"

#TODO:calculate mask to exclude empty area? Please set None for now.
mask = None

#Size of training cubes, this size should be divisible by 2^unet_depth
cube_sidelen = 96

#size of cubes to impose missing wedge. Should be same or larger than size of cubes
cropsize = 128

#number of cubes generated for each (sub)tomos.
#Because each (sub)tomo rotates 16 times, the actual number of cubes for trainings should be ncube*16
ncube = 256

#------------------------------------------------------------------------------------------
# train settings

#Number of epoch for each iteraction
epochs = 10

#size of the minibatch
batch_size = 4

# step per epoch. A good estimation of this value is number is
# (sub)tomos * ncube * 16 / batch_size *0.9
steps_per_epoch = 800

#Add noise during training, Set None to disable noise reduction
noise_folder = None

#level of noise STD(added noise)/STD(data)
noise_level = 0.5

#Drop out rate to reduce overfitting
drop_out = 0.3

#number of convolution layer for each depth
convs_per_depth = 4

#kernel for convolution layer
kernel = (3,3,3)

#depth of Unet
unet_depth = 4

#Sometimes batch normalization may induce artifacts for extreme pixels in the first several iterations.
#those could be restored in further iterations
batch_normalization = False

#Normalize the 5% and 95% pixel intensity to 0 and 1 respectively
#If this is set to False, normalize the input to 0 mean and 1 standard dievation
normalize_percentile = True


#---------------------------------------------------------------------------------------------
#predict settings

#if datas_are_subtomos the predict cube size and crop size are both size of subtomo
#cropsize should be larger than cubesize. Often 1.5 times of cubesize
predict_cubesize = 96
predict_cropsize = 144

predict_batch_size = 4


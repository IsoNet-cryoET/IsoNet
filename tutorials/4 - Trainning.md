
## Training
This process iteratively fills the missing wedge information by training deep neural networks using the same tomogram whose missing wedge artifacts were added to other directions. The sub-tomograms are first sampled from tomograms with the constrain of the mask(optional), generating the dataset for network training. Then iterative refinement will be carried, including training network, correcting sub-tomograms and updating training dataset.

For detail instruction, please run the following command.*mwr_cli.py train --help*

### General Parameters
#### --input_dir
Directory of tomograms from which subtomos are sampled; Those tomograms are suggested to have similar content and imaging conditions. The format of the tomograms\ have to be .mrc or .rec. 
#### --datas_are_subtomos, --subtomo_dir
When your inputs are subtomograms, you should set the *datas_are_subtomos* to **True** and the *subtomo_dir* to the path of your subtomograms folder. The training dataset will be generated from provided subtomograms instead of sampling from entire tomograms.
#### --iterations 
One iteration contains 3 steps: data preprocessing, network training, predicting. When *iteration* >1, the program will train the network iteratively. The model will be refined in current iteration based on previous prediction results. However, it doesn't mean that the more ierations the better. In practice, 50 is sufficient.
#### --pretrained_model
The path of the pretrained neural network model. MWR will initialize model parameters from the pretrained parameters if a pretrained neural network model with the same configuration is provided here. This model can be generated from your previous training or download.
#### --noise_dir
Add noise during training, Set None to disable noise reduction. 
#### --mask_dir
The path to your mask folder. For each tomogram in **the input_dir**, the software will match corresponding mask by file name. Usually, you should use mask when your tomograms contain large empty area. To avoid sampling large amount of empty sub-volumes, a mask which exclude the empty area is needed. User could either use the *mwr_cli.py make_mask* to generate the mask or get it from other programs as long as the volume dimension is consistent with your tomograms.
#### --continue_training, --continue_iter
Continus previous training? Every model after an iteration will be saved.When *continue_training* is set True, iterative training will start from the previously trained model of *continue_iter* 
#### --cube_size , --crop_size
sub-tomograms with size cube_size^3 are extracted randomly from the input tomograms during data preprocessing. If masks are applied, the center of the sub-tomograms should be within the masked areas. Those sub-tomograms are then rotated and cropped to the size serve as the inp.Sub-tomograms are applied with missing wedge mask, then cropped to the crop_size size to serve as the output of the network. Usually the crop_size is 1.5 time large as the cube_size. By default, each sub-tomograms are rotated 16 times making 16 data pairs. Those generated input-output data pairs store in the data folder and will be used in the following training. 



### Advanced Options


#### -- preprocessing_ncpus
Number of CPU will be use in the data preprocessing step which can only be performed with cpu.

#### --gpuID 
ID of gpus will be used in the network training. i.e. 0,1,2,3
Users can find information of their available GPUs through commond: **nvidia-smi**
#### --epoch, --batchsize and --step_per_epoch
The training step is divided into several epochs. Each epoch will traverse through the data set.
The data pairs are grouped into batches to feed into each epoch.
It is recommended that the batchsize times the step_per_epoch approximately equals the number of the data pairs. If you are using multiple GPUs, the batchsize should be divisible to the number of GPU. The trained neural networks are saved in the results folder named model_iterxx.h5 by default. 
#### --ncube
Number of cubes generated for each (sub)tomos. Because each (sub)tomo rotates 16 times, the actual number of cubes for trainings should be ncube times 16.
#### --noise_level，--noise_start_iter，--noise_pause
The *noise_start_iter*  defines when the noise will be added to the training data. After every the number of *noise_pause* iterations, the overall artificial noise level, which is the ratio of standard deviation of the noise volume compare to the original data, will increase by *noise_level* .Loss will increase if noise was added.


## Parameters tuning
The following parameters need to be tuned with respect to user's data and hardware.
--Iteration: the number of iterations that will be used to correct missing wedge. In each iteration, three steps will be performed: data preprocessing, network training, and prediction.
--cube_size and --crop_size：the sub-tomograms with size **cube_size** are extracted randomly from the tomograms during data preprocessing. If masks are applied, the center of the sub-tomograms should be within the masked areas.
--cube_size：the sub-tomograms with size **cube_size** are extracted randomly from the tomograms during data preprocessing.Those sub-tomograms are then rotated and cropped to the size serve as the input. Those same sub-tomograms are rotated, added missing wedge and then cropped to the same size to serve as the output of the network. Usually the **crop_size** is 1.5 time larger than the **cube_size**.
  By default, each sub-tomograms are rotated 16 times making 16 data pairs. Those generated input-output data pairs store in the **data** folder and will be used in the following training. The preprocessing step can only be performed with cpu, which specified in the parameter **preprocessing_ncpus**.
The training step used GPU to train the neural network. The ID of GPU is can be assigned in **gpuID**. The training step is divided into several **epochs**. Each epoch will traverse through the data set. The data pairs are grouped into batches to feed into each epoch. It is recommended that the **batchsize** times the **step_per_epoch** approximately equals the number of the data pairs. If you are using multiple GPUs, the **batchsize** should be divisible to the number of GPU. The trained neural networks are saved in the **result_dir** folder named model_iterxx.h5. 
The prediction step performs the missing wedge correction of each sub-tomograms. Similar to the training, the sub-tomograms are grouped in to batches **predict_batch_size**.  The predicted sub-tomograms will be used for processing for the subsequent iterations.
 
## That’s it
The resulting **model_iterxx.h5** is the network to be used in the next module: Missing wedge correction of the entire tomogram.

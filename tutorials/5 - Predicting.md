
## Predict
This step applies the already trained network model to the tomogram in order to correct the missing wedge of that tomogram. 
 
The trained network can either be the network downloaded from the pretrained network in https://drive.google.com/file/d/1JFGxmW2V4EYUJqBVKoG_9Cd7fM77fmKc/view?usp=sharing or the network model trained with the user’s data, named model_iterxx.h5 in the results folder.

## General Parameters
#### mrc_file, output_file
The input mrc_file can be the exact tomograms used for training or other tomograms with the similar content and imaging condition. Predicted tomograms will be saved as output_file
 
#### model
The model file can be model trained on the corresponding training dataset from input tomograms or other pretrained model. Because the neural networks learn to restore information based on the features in training data, similar tomograms can share common trained networks.
Model file should be in .h5 format
 
## Advanced Parameters
#### --batchsize
The batch size of the cubes grouped into for network predicting. Greater batchsize will save more predicting time. It should be tuned compatible with your GPU memory
 
#### --gpuID
ID of gpus will be used for predicting i.e. 0,1,2,3 Users can find the information of their available GPUs through command: nvidia-smi
 
#### --cube_size, --crop_size
To fit the tomogram into the GPU memory, the entire tomogram is divided into multiple tiles for the missing wedge correction and using the overlap tile strategy to prevent the artifact during the montaging the tiles. To implement this strategy, the *crop_size* should be larger than the *cube_size*. The *cube_size* and *crop_size* is suggested to be consistent with the training settings.
 
#### --norm
If norm is set True, tomograms will be normalized by percentile, which scale the sub-tomograms in a range approximately from 0 to 1. If this is set False, the sub-tomograms will be normalized to have a mean of zero and a standard deviation of 1.


```bash
%%bash
mwr_cli.py predict path_to_tomogram  path_to_output_tomogram  path_model --gpuID 0,1,2,3
```

    Process is terminated.


### Batch predicting



```python
mwr_cli.py predict tomoset corrected_folder path_model --gpuID='0,1,2,3' --cube_size=80 --crop_size==128
```

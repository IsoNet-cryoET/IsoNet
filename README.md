## Installation
python version at least 3.5 is required. If you download the package as a zip file from github, please rename the folder IsoNet-master to IsoNet.

1.  IsoNet relies on Tensorflow with version at least 2.0

Please find your cuda version, cuDNN version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu. 

For example, if you are using cude 10.1, you should install tensorflow 2.3:
```
pip install tensorflow-gpu==2.3.0
```

2.  Install other dependencies

```
pip install -r requirements.txt
```
3.  Add environment variables: 

For example add following lines in your ~/.bashrc
```
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
```
4. Open a new terminal, enter your working directory and run 
```
isonet.py check
```

Tutorial data set and tutorial videos are on google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp

# FAQ:
## 1. IsoNet refine raise OOM error.

This is caused by the insufficient GPU memory.
The soluitions are:
1. Specify a smaller batch\_size or use more(powerful) GPUs. The default batch\_size is 4 if you use one GPU, otherwise the default batch\_size is 2 times the number of GPU. Please note the batch_size should be multiple of number of GPUs, so the minimum batch\_size is the number of GPU you are using.
For example, if you have one GPU and get OOM error, please reduce the batch\_size to 1 or 2; If you use 4 GPUs and get OOM error, please reduce the batch\_size to 4.

2. Refine with a smaller cube\_size (not recommanded).

## 2.  IsoNet extract ValueError: a must be greater than 0 unless no samples are taken
This could be due to the tomogram thickness is smaller than the size of subtomograms to be extracted. Please make your tomogram thicker in this case.

## 3. Can not see significent improvement after processing with IsoNet
IsoNet is kind of conservative in adding information into missing wedge region. If it can not find reasonable prediction, IsoNet may simply returns the origional tomograms back to you. 
However, there are some ways to increase your success rate.
1. IsoNet performs better in high contrast tomograms. That means it will be helpful to tweak the parameters (especially snrfalloff) in CTF deconvolution step to make increase the weight of low resolution information. Or trying with the data acquired with phaseplate first. As far as we know, phaseplate data will always give you good result.

2. Missing wedge caused the nonlocal distributted information. You may observed the long shadows of gold beads in the tomograms, and those long shadows can not be fully corrected with sub-tomogram based missing correction in IsoNet, because the receptive field of the network is limitted to your subtomogram. This nonlocal information makes it particular difficult to recover the horizontal oriented membrane. There are several ways to improve. **First**, training with tomograms with larger size, the default cube size is 64, you may want to increase the size to 96 or 128, however this may lead to the OOM error Please refer to FAQ #1 when you have this problem. **Second**, bin your tomograms more. Some times we even bin our celluar tomograms to 20A/pix for IsoNet processing, this will of course increase your network receptive field, given the same size of subtomogram. Do not worry too much about binning, we found 40A resolution is sufficient for visualization of celluar tomograms. 

3. Increase batch_size and reduce the learning rate when the loss is fluctuating or stagnant. Both of the strategy is to make sure the traning is stable. Because the subtomograms are noisy, the gradient of one batch may diverge from the "true" gradient. To reduce the effect of the noise, you can reduce the learning rate so that the each noisy image may not affect too much in updating networks. Increasing the batch size has similar effect, and it will also make the batch normalization layer more efficient. However, the batch size can not be too large due to the memory limit of the GPU. Both the choice of batch size and learning rate are emperical and may depend on the data. 
## 4. Can not create a good mask during mask generation step
The mask is only important if the sample is sparsely located in the tomograms. And the mask do not need to be perfect to obtain good result, in other words, including many empty/unwanted subtomos during the refinement can be toralated. 

To obtain a good mask, the tomograms should have sufficient contrast, which can be achieved by CTF deconvolution. User defined mask can also be supplied by changing the mask_name field in the star file. Alternately, you can also use subtomograms extracted with other methods and skip the entire mask creation and subtomograms extraction steps.

If you want to exclude carbon area of the tomograms, you can try the new mask boundary feature in version 0.2. It allows you to draw a polygon in 3dmod so that the area outside the polygon will be excluded.
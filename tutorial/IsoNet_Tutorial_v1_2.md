# IsoNet Tutorial

[toc]



## Introduction

**IsoNet** stands for for ISOtropic reconstructioN of Electron Tomography. It trains deep convolutional neural networks to reconstruct meaningful contents in the missing wedge for electron tomography, and to increase signal-to-noise ratio, using the information learned from the original tomogram. The software requires tomograms as input. Observing at about 30A resolution, the IsoNet generated tomograms are largely isotropic. 

### IsoNet Design

**IsoNet** contains 3 main modules to achieve the missing wedge correction: mask generation, model refining, and predicting. 

All commands are based operations on **.star** text files which include description and path of data and parameters for tasks mentioned above. For detailed descriptions of each module and .star preparation and manipulation please refer to the individual chapters.

Users can choose to utilize IsoNet through either GUI or command-lines. **Step by step examples** is in the next chapter. Also, the video tutorial, which handles tomograms of virus particle sample, can be found [here](https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp).

### Installation

Follow the instruction here [IsoNet/README.md at master · Heng-Z/IsoNet (github.com)](https://github.com/Heng-Z/IsoNet/blob/master/README.md)



## Examples

The following two quick exmaples describe the basic steps of running the program and generating missing wedge corrected tomograms using both command line and GUI. 

1. [Here](./Example1.md) is the first example which handles a tomogram of synapse structure which is crowding and heterogeneous. 

<img src="figures/pp676_demo.png" alt="pp676_demo" width="600" />

2. [Here](./Example2.md) is the second example which processes tomograms of HIV virus particles. These tomograms are relatively empty and noisy and the content is homogeneous. An additional step (CTF deconvolution) is performed.

<img src="figures/fig1.png" width="700" />

For a detailed description of each command in these two examples, please refer to **Individual tasks**. Among the following tasks, only refinement and prediction steps are computational extensive and require GPU for acceleration.



## Individual tasks

### 1. Prepare tomograms and STAR file

IsoNet uses star file format to store information of tomograms. This
file can be prepared using **isonet.py prepare_star** command prior to
the subsequent processing. To do so, users should prepare a folder
containing all tomograms. Binning the tomograms to pixel size of about
10 A is recommended since the target z-axis resolution should be about
30A. Too large (>20A) or too small
(<5A) pixels might reduce the efficiency of IsoNet
network training. The default **pixel_size** parameter is 10A. We
typically use a folder containing 1 to 5 tomograms as input.

The input tomograms can be either reconstructed by SIRT or WBP
algorithm. The tilt axis is the y-axis and recommended tilt range is -60 to
60 degrees, without x-axis tilt, while other tilt ranges might also
work. The tilt series can be collected with any tilt schemes, continuous,
bidirectional or dose-symmetric.

If you do not want to perform CTF deconvolution, especially when the
tomogram is acquired by phase plate or the tomogram is CTF corrected,
the **defocus** can be left as 0. Otherwise, you can use the **defocus**
parameter to set one defocus value for the tomograms. This value should
be the defocus value of the zero tilt image. We do not consider defocus
variation across different tilted images in this version of IsoNet,
since it does not target high resolution currently. When you have
multiple tomograms in the folder, the **defocus** parameter (in
angstrom) for each tomogram should be adjusted in the star file with
your text editor, after the tomogram star file has been generated with
**isonet.py prepare_star** command.

### 2. CTF deconvolve

Given the defocus values in the tomogram star file, CTF deconvolution
can be performed by applying a Weiner filter to each tomogram. This step
not only reduces the CTF artifact but also enhances the low-resolution
signal so that it is easier to train the network. This step can be
skipped for tomograms acquired with a phase plate. This step is similar to
the CTF deconvolve in Warp software.

#### Deconvolution parameters

Two parameters, **snrfalloff** and **deconvstrength**, are worthy to
tune in this step so that the output tomograms have visually highest
contrast. If these parameters are not set in the command, the values in
the *star* file will be used; If the star file does not contain these
parameters, default 1.0 will be used for both parameters. The effect of
these two parameters is shown in the following figures. You can also
specify **tomo_index** (e.g. 1,3-4 ) so that only the specific tomogram
or tomograms will be processed. Another parameter **hipassnyquest**
applies a high pass filter at very low frequency, changing this
parameter might be helpful if the tomograms are too blurry. 



<img src="figures/snrfalloff.png" width="800">

<p style="text-align: center; font-size: 11pt"> 2D slices of CTF deconvolved tomograms with different deconvstrength parameters. 
			Left: deconvstrength=0.5; middle: deconvstrength=1;right:deconvstrength=1.5 </p>

<img src="figures/decovstrength.png" width="800">

<p style="text-align: center; font-size: 11pt"> 2D slices of CTF deconvolved tomograms with different snrfalloff parameters. Left: snrfalloff=0.5; middle: snrfalloff=1;right:snrfalloff=1.5

#### Parallel processing

Multiple CPUs can be used for CTF deconvolution, the number of CPUs is set as
parameter **ncpu**, the default value for **ncpu** is 4.

When your computer has enough memory. Please keep **chunk_size** as
None. Otherwise, You can let the IsoNet program crop the tomogram into
multiple chunks for multiprocessing and assembly them into one. The
**chunk_size** defines the size of an individual chunk. This option may
induce artifacts along the edges of chunks. When that happens, you may try
a larger **overlap_rate**.

### 3. Generate mask

To obtain a training dataset, sub-tomograms are randomly extracted from
tomograms. However, when the sample in a given tomogram is sparsely
distributed, most of the extracted sub-tomograms will not contain
meaningful information. Therefore, the performance of network training
might be reduced, although in most cases, you can still get a reasonable
result from training without masks.

To help with getting a more meaningful training dataset, we introduce the
**make_mask** module, which created three types of mask: 1 Pixel
intensity mask, 2. Standard deviation mask, 3. mask that crops out top
and bottom parts of tomograms

The final mask created by the command is the **intersection** of these
three types of masks.

#### Pixel intensity mask

This type of mask will mask out empty areas based on their relatively
low local pixel density. A maximum mask will first suppress the noise
with a 3D Gaussian filter and then apply a 3D sliding window
maximum-density filter to the tomograms. The window size of the maximum
filter is defined by **patch_size** parameter. This size can be
increased if the tomograms are too noisy.

In this filter tomogram, the areas with relatively smaller density
values will be deemed as empty space and can be excluded with parameter
**density_percentage**, ranging from 0 to 100, meaning the only include
this percentage of pixels in the mask.

Usually, a lower **density_percentage** value should be used when having
tomograms with sparsely distributed samples. However, this type of mask
does not work well when the tomograms do not have uniform backgrounds,
e.g. darker on one side of the tomogram. When you don't want to use this
mask, set the **density_percentage** value to 100.

#### Standard deviation mask

Recognizing that the \"empty\" regions of the tomograms are often areas
with low standard deviation (STD), this standard deviation mask is
designed to exclude the low STD areas.

To do so, we calculated the STD of a volume centered at the evaluating
pixel (local STD). The size of the volume to measure local STD is
defined by the parameter **patch_size**, this parameter is also used to
define the size of max filter in pixel intensity mask.

All the pixels with STD ratio larger than the **std_percentage**% of all
pixels will be included in the mask. When you don't want to use this
mask, set the **std_percentage** value to 100.

<img src="figures/mask.png" width="800">

<p style="text-align: center; font-size: 11pt">XY slices of tomograms and corresponding masks. Both using density_percentile=50 and std_percentile=50 </p>



#### Crop out the top and bottom parts of tomograms

Sometimes the top and bottom parts of tomograms are devoid of sample.
**z_crop** parameter is designed to exclude the top and bottom regions
of tomograms along the z-axis. For example, **\--z_crop 0.2** will mask out
both the top 20% and bottom 20% region along the z-axis.

### 4. Extract

This step randomly extract the subtomograms from the tomograms in the
input tomogram star file. The output of this command is a folder
containing all the subtomograms and a star file containing the
information of those subtomograms.

The number of subtomograms to be extracted for each tomogram should be
written in the tomogram star file in the column of (\_rlnNumberSubtomo).
Users can modify this number in the star file. Ideally, the smaller the
masked area, the smaller this number should be. Usually total 300
subtomograms are sufficient for refinement.

If mask files are provided in the input tomogram star file, the centers
of the subtomograms are always inside the mask regions. If CTF
deconvoluted tomograms are provided in the tomogram file, those
tomograms will be used for the subtomogram extraction unless the
**use_deconv_tomo** parameter is set to False.

The parameter **cube_size** is the size of the cubic volume for
training, not the size of extracted subtomograms. The actual size of the
extracted subtomograms, which is defined as **crop_size**, is by default 16 + cube_size. The **cube_size**
should be divisible by 8 and is usually limited by the GPU memory. 64 or
96 is often a good estimation for **cube_size**. If you encountered
an out-of-memory (OOM) problem during network training, reducing this value
is one of the choices.



#### Prepare *star* file of subtomogram from particles

This step allows generating a *star* file from a folder containing only
subtomogram files. The subtomogram files should be in mrc format, with
extension of \".mrc\". The generated star file can be used as input for
refinement.

This step is not in the main workflow of IsoNet, but might be helpful if
you already have subtomograms extracted through other
softwares. For example, if you are only interested in processing
subtomograms of a particular protein, you can manually pick and extract
them, then prepare star file with this command. We also recommend these
subtomograms are downing-scaled to 5-15 A/pixel and CTF deconvoluted if
those are not acquired with phase plate.

This command works as follows:

``` {.bash language="bash"}
isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size] 
```

Where \"folder_name\" is the folder containing subparticles. The default
output star file is \"subtomo.star\".

And the \"cube_size\" is the size of the cubic volumes used for training.
This value is smaller than the size of subtomograms and should be
divisible by 8, eg. 64, 96. If this value isn't set, \"cube_size\" is
automatically determined as int(subtomo_size / 1.5 + 1)//16 \* 16

### 5. Refine

This process iteratively trains neural networks to fill the missing
wedge information using the same tomograms whose missing wedge artifacts
were added to other directions. The denoising module can also be enabled
in this step, making the network capable of both reducing noise and
Recovering missing wedge. After refinement, the subtomograms neural network
model in each iteration are saved in the results folder. The network
models with suffix of \".h5\" can be used for the prediction step.

#### Optimizing computation resources

At the beginning of each iteration, IsoNet will process the
subtomograms, such as rotating, cropping, and applying missing wedge.
Only CPUs are used for those processes. The parameter
**preprocessing_ncpus** is used to define how many CPU cores are used
for this preparation step.

Users have to specify **gpuID**, e.g. 0,1,2,3, for the network training,
so that 3D volumes will be distributed across those GPUs. Information of
available GPUs can be found through the command: *nvidia-smi*. In general,
Using more GPUs reduces GPU memory requirement for each GPU and current
network training can not be performed across multiple computer machines.

The **data_dir** is the folder that temporally stores the training
and test data pairs. The files in that folder will be updated every
iteration. Setting it to a faster drive such as SSD or memory file
system will presumably increase the speed of network training, though
not fully bench-marked on the developer's side.

#### Optimizing training speed

One **iteration** of refinement contains three steps: training data
preparation, network training, and subtomogram prediction. The model will
be refined iteratively based on previous prediction results. In
practice, a total of 10 to 20 iterations is typically used for refinement
without denoising. Please refer to the denoising section for more details.

The training step is divided into several **epochs**. Each epoch will
traverse through the randomly shuffled data set. The default value 10
for the number of epochs is usually sufficient. Training data pairs are
grouped into batches to feed into each epoch. The **batch_size** should
be divisible by the number of GPUs so that the data can be distributed
into multiple GPUs. If you are using multiple GPUs and **batch_size** is
not set, the default value is two times the number of GPUs. If you are
using a single GPU, the default **batch_size** is four. We tested
**batch_size** of 4-12 can result in a good performance, too large
**batch_size** might lead to OOM error. **steps_per_epoch** defines how
many batches are to be processed in one epoch. A value between 100 to
300 are recommended for **steps_per_epoch**. If this value is not set by
Users, the default **steps_per_epoch** is min(number_of_subtomograms \* 6
/ batch_size , 200)

#### Denoising

The **noise_start_iter** parameter defines when the denoising will be applied to the training. Usually, it is the iteration in which undenoised
training the expected to be converged. The **noise_level** is the ratio
of the standard deviation of the additive noise compare to the original
data. Once the denoise was applied, the mean absolute error loss of the
training will be increased. If the **noise_level** is too high, the
training might fail, as indicated by the extremely large losses.

You can set both **noise_start_iter** and **noise_level** with multiple
values. So that you can gradually increase the **noise_level** during
the training. We recommend you use multiple values of the
**noise_level** and **noise_start_iter**.

For example, in the following command, the **iteration** parameter is
set to 30, **noise_level** is set to 0.1,0.2, and **noise_start_iter**
is 11,21. Then the first 10 iterations are trained without denoising,
11-20 iterations are trained with noise level 0.1, 21-30 iterations are
trained with noise level 0.2. You can use the neural network model from
iteration 10, 20, 30 to predict the same tomogram and distinguish which
level of denoising is best for your tomograms.

``` {.bash language="bash"}
isonet.py refine subtomo.star --iter 30 --noise_level 0.1,0.2 --noise_start_iter 11,21
```

#### Network structures

IsoNet allows users to modify the network structures by the input
arguments. For example, it might be useful to increase or decrease the
size of the network by increasing or decreasing **unet_depth**, though this is
not recommended unless users want to venture into the performance of
different networks. Another parameter that decides the complexity of the neural network is **filter_base**. It determines the scale of the number of feature channels and has been set to 32 by defaults. Increasing it could lead to a better result at the cost of a longer training time. Please note that these values will be ignored when using **pretrained_model**.

If **normalize_percentile** is set True, tomograms will be normalized by
percentile, which scales the sub-tomograms in a range approximately from
0 to 1. If this is set False, the sub-tomograms will be normalized to
have a mean of zero and a standard deviation of 1.

#### Continuing using the previously trained network

If you want to continue with a model from previous iterations of refine,
you can specify the **continue_from** argument, which takes the '.json'
file that is generated at each iteration.

For example:

``` {.bash language="bash"}
isonet.py refine subtomo.star --continue_from ./results/refine_iter30.json --gpuID 0,1,2,3 --iterations 50 
```

If you already have a trained network model (with a file extension of
.h5), instead of '.json' file , you may want to choose
**pretrained_model** option:

For example:

``` {.bash language="bash"}
isonet.py refine subtomo.star --pretrained_model ./pretrained_model.h5 --gpuID 0,1,2,3 
```

This command enables using your pretrained model to predict the
subtomograms of the first iteration. Starting with the second iteration,
you are refining this model using the subtomograms in the subtomo.star

### 6 Predict

This module applies the trained network model to tomograms to restore
the information in the missing wedge region of tomograms. It takes the
tomogram star file, which can be generated with **isonet.py
prepare_star** command, and a trained network model (with .h5 file
extension) as input. The input tomograms in the tomogram star file are
typically the exact tomograms used for training or other tomograms with
similar sample and imaging conditions. If the network is trained with
CTF deconvolved tomograms, the tomograms used for predict should also be
CTF deconvolved.

This step is much faster than refine step. **gpuID** defines which
GPU(s) will be used for predicting, e.g. 0,1,2,3. If this parameter is
not set, CPU will be used for prediction, which could take much longer
time than using GPU.

To fit the tomogram into the GPU memory, one tomogram is divided into
multiple tiles for the missing wedge correction and using the overlap
tile strategy to prevent the artifact during montaging the tiles. To
implement this strategy, the **crop_size** should be larger than the
**cube_size**. The **cube_size** and **crop_size** are suggested to be
consistent with the training settings. If **crop_size** is not large
enough, you may observe artifacts of grids between the adjacent tiles.

The **batch_size** defines the number of subtomograms grouped
together for network predicting, this value should be divisible by the
number of GPU. Larger **batch_size** will save more predicting time but
occupy larger your GPU memory. **normalize_percentile** should be the
same as that parameter in refine.

``` {.bash language="bash"}
isonet.py predict tomogram.star path_to_network_model --gpuID 0,1,2,3 --cube_size 80 --crop_size 128
```



## GUI

This section briefly described IsoNet graphic user interface (GUI). You
can watch our [tutorial video](https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp) for the details on how to use GUI.

The GUI can be started by the following command:

``` {.bash language="bash"}
isonet.py gui
```

<img src="figures/gui_1.png" width="600">

This software mainly have 3 pages, they are **Preparation**,
**Refinement** and **Prediction**.

1.  Preparation includes the preprocessing steps to prepare the dataset to
    be trained in the later Refinement step.
    -   Deconvolve CTF: increase the contrast of tomograms with no VPP
        applied

    -   Generate mask: only mask out the region of interest in the
        tomogram

    -   Subtomograms extraction: generate the training dataset

2.  Refinement will train a neural network to learn how to correct the
    missing wedge effect.

3.  Prediction help users to generate corrected tomogram based on the model they trained in the Refinement step.

IsoNet takes the **.star** file as its input file, which follows the
convention from *Relion*. The GUI will automatically read the
**tomograms.star** as default if it exists in the current folder.

In the preparation tab, an input table can help users create or load their own
**.star**. One can click insert to add a new row into the table. For
tutorial dataset of HIV, you need to specify MicrographName
(reconstructed tomogram), pixel size, estimated defocus value for your 0
degree image. You can select one row by clicking the index of the row on
the left-most region. After being selected, you can duplicate it or delete it
by clicking the insert or Delete button on the right. '3dmod view' helps
users to visualize selected tomograms and/or masks.

Further operations of IsoNet GUI are intuitive. Please refer to the [video tutorial](https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp) or the [first example](Example1.md).




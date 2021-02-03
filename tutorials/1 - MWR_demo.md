
# This is a step-by-step tutorial of the training in MWR. 
 
(ISNet ISotropic reconstructioN of Electeon Tomography.)
This notebook describes the requirement to run the program and generate missing wedge corrected tomogram from the demo data.
MWR contains four modules to achieve the missing wedge correction: noise generation, mask generation, training, and predicting. For detailed descriptions of each module please refer to the individual tutorial notebook.

Now we assume you already installed the MWR software. **For note on the installation, please refer to xx.** To evaluate whether your installation is successuful you can type  


```bash
%%bash
mwr_cli.py check
```

    MWR --version 0.9.9 installed


    Using TensorFlow backend.


## Demo data 
The tutorial dataset is an 3D cryo electron tomogram of a hippocampal neuronal synapse (1). The tomogram was reconstructed using weight back projection.
 
 
The tutorial data can be downloaded from https://drive.google.com/drive/folders/1wdO4NozmWMA7bPa-RTfzrZRgAmKwTcR3?usp=sharing

Create a new tomo folder and move your downloaded demo data into your tomo folder.

When you open the pp676-bin4-wbp.rec file in xyz view in IMOD, you may see what similar to the following images. The 3D volume is a bit noisy and the quality is worse in yz and xz plane.


```bash
%%bash
mkdir tomo
mv demo_synapse.mrc tomo/
```


```python
from IPython.display import Image
Image(filename="./images/origional_data.png")
```




![](./images/original_data.png)



## Download pretrained model
MWR can train networks with initial parameters obtained from a pretrained model. This step will speed up your training.

You can choose the model trained on the dataset which is most similar to yours.

You can download the pretrained model of the neuron dataset from https://drive.google.com/file/d/1JFGxmW2V4EYUJqBVKoG_9Cd7fM77fmKc/view

## Generate mask 
To exclude the areas that devoid of sample, we apply a binary mask to each tomogram. This step will improve the performence.

You can skip this step if you just want a quick go-through


```bash
%%bash
mkdir mask 
mwr_cli.py make_mask ./tomo/demo_synapse.mrc ./mask/demo_synapse_mask.mrc --percentile=99 --side=10
```

    Gaussian_filter
    maximum_filter
    mask generated


## Generate training noise
To help accelerate the missing wedge information retrieval, some artificial noise, which mimics the noise pattern in original tomogram, can be added to the input sub-tomograms. Noise cubes are 3d arrays and also exhibit missing wedge effect.
The size of noise cubes have to be the same as sub-tomograms' in the training data.


```bash
%%bash
mwr_cli.py generate_noise ./noise64 1000 64 

```

    noise generated 


## Training
This process iteratively fills the missing wedge information by training deep neural networks using the same tomogram whose missing wedge artifacts were added to other directions. The sub-tomograms are first sampled from tomograms with the constrain of the mask(optional), generating the dataset for network training. Then iterative refinement will be carried, including training network, correcting sub-tomograms and updating training dataset.

Using following command, it will take 1 hour for four 1080ti to complete refinement with the pretrained model.


```bash
%%bash
mwr_cli.py train --input_dir ./tomo --mask_dir ./mask --pretrained_model neuron_pretrained.h5  --noise_dir noise64 --iterations 6 --steps_per_epoch 200  --nucbe 300 --noise_level 0.1 --noise_start_iter 0 --noise_pause 3 --epochs 8 --batch_size 8 --gpuID '0,1,2,3'
```

    training


Omit the pretrained_model option if you don't have one, but it will take 8 hours to finish the refinement.


```python
mwr_cli.py train --input_dir ./tomo --mask_dir ./mask --noise_dir noise64 --iterations 30 --steps_per_epoch 200  --nucbe 300 --noise_level 0.1 --noise_start_iter 15 --noise_pause 3 --epochs 8 --batch_size 8 --gpuID '0,1,2,3'
```

## Predicting
The network models will be saved after each iteration during refinement. With the network models generated in results folder, you select one and apply it to your entire tomograms.


```bash
%%bash
mwr_cli.py predict ./tomo/demo_synapse.mrc demo_synapse-iter40.mrc ./results/model_iter40.h5
```

    successfully predicted


## that's it 
we now have the missing wedge corrected tomogram named demo_synapse-iter40.mrc



```python
from IPython.display import Image
Image(filename="./images/demo_synapse_corrected.png", width=500,height=500)
```




![](./images/demo_synapse_corrected.png)



 
*1 Tao, C. L. et al. Differentiation and Characterization of Excitatory and Inhibitory Synapses by Cryo-electron Tomography and Correlative Microscopy. J Neurosci 38, 1493-1510, doi:10.1523/JNEUROSCI.1548-17.2017 (2018).*

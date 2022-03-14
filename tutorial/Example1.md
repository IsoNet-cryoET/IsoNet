### A step by step example

This example processes tomogram of **synapse structure** through **IsoNet**.

**Following is the command-line usage of** **IsoNet**

0. Demo data

Prepare working directory like following; Data used for demonstration can be found here

```
isonet_demo/
└── tomofile
    └── pp676-bin4-5i-demo.rec
```

1. Prepare STAR file 

command line:

```
isonet.py prepare_star tomofile/ --pixel_size 18.12 --number_subtomos 300
```

Default file name of STAR file will *tomograms.star* ; Contents in this file can be modified through text editor.

2. Deconvolution

This step is unnecessary for tomographs accquired with phase plate.

3. Generate mask

To exclude the areas that devoid of sample, we apply a binary sampling mask to each tomogram. This step will improve the quality of training dataset.

Since this demo tomogram has a bin-factor of 4, a smaller gaussian filter can smooth out noise and keep the fine structure. We set *patch_size* to 2. 

Usually, the top and bottom (along the z axis) region is in lack of content. We mask out the top 15% and bottom 15% region by setting *z_crop* to 0.15.

```bash
isonet.py make_mask tomograms.star --z_crop 0.15 --patch_size 2
```

The output mask will be placed in the **mask** by default.

4. Extract subtomogram

This step extracts small volumes (here we also call subtomograms) from big tomogram , with the sampling mask generated in the preivous step.

```
isonet.py extract tomograms.star
```

The default subtomogram folder is **subtomo**. And **subtomo.star** file is the corresponding STAR file of subtomograms.

Up to this step, our workspace is like this:

```
isonet_demo/
├── mask
│   └── pp676-bin4-5i-demo_mask.mrc
├── subtomo
│   ├── pp676-bin4-5i-demo_000000.mrc
│   ├── pp676-bin4-5i-demo_000001.mrc
│   ├── pp676-bin4-5i-demo_000002.mrc
│   ├── pp676-bin4-5i-demo_000003.mrc
		.
		.
		.
│   ├── pp676-bin4-5i-demo_000298.mrc
│   └── pp676-bin4-5i-demo_000299.mrc
├── subtomo.star
├── tomofile
│   └── pp676-bin4-5i-demo.rec
└── tomograms.star
```



5. Refinement

This process iteratively fills the missing wedge information by training deep neural networks with subtomograms. The default refinement parameters are great for most cellular tomograms. Here, we only set the ID of GPU and the results directory name where the neural network models are saved during iterative refinement.

```
isonet.py refine subtomo.star --gpuID 0,1 --result_dir demo_results
```

The training time depends on the GPU performence. This example take **7 hours** on two Nvidia gtx1080 cards for 25 iterations.

6. Predict

All trained neural-net models after each itereation are stored in the results directory (**demo_results** in this example ) . Predict will imploy one of these models (typically models after 25~30 iterations, corresponding to denoise level between 0.15~0.20) to do missing wedge correction to the original tomogram (pp676-bin4-5i-demo_mask.mrc). In this demonstration, we use the model after 25 iteration.

```bash
isonet.py predict tomograms.star demo_results/model_iter25.h5 --gpuID 0,1
```

The predict time comsumption is much less than refinement step. This tomogram will take less than **2 minutes** to predict using two Nvidia gtx1080 cards.



**Following is the GUI usage of IsoNet**

0. Open a terminal window, type following to lauch the GUI

```
isonet.py gui &
```

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui1.png" alt="image-20220207001846094" style="zoom:30%;" />

1. Add tomograms

Click *insert* on the upper-right panel and then click *None* in the *MicrographName* colume, select one tomogram.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_addtomo.png" alt="image-20220207002107926" style="zoom:30%;" />

Set the pixel size and the number of subtomogram to be extracted from this tomogram.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_settomo.png" alt="image-20220207002456475" style="zoom:33%;" />

2. Generate mask

Set proper parameter for producing mask.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_genmask.png" alt="image-20220207002807345" style="zoom:33%;" />



And click **Generate Mask ** button.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_clickmask.png" alt="image-20220207002915179" style="zoom:33%;" />

We can view the produced mask via 3dmod: Click the small index  '1'  at the beginning of the row in the tomogram table and click **3dmod view**

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_viewmask.png" alt="image-20220207003308839" style="zoom:33%;" />

Check the mask you generate and adjust parameter if necessary.

3. Extract subtomogram

Click **Extract**

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_clickextract.png" alt="image-20220207002915179" style="zoom:33%;" />

4. Refinement

Click the **Refinement** tab at the upper most panel and set the GPU ID and results folder name.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_refinetab.png" alt="image-20220207003550613" style="zoom:33%;" />

Click **Refinie** button below to start iterative refinement.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_clickrefine.png" alt="image-20220207003812744" style="zoom:33%;" />

If you want to run refinement step on a remote machine or cluster using command-line, you can check the 'only print command ' box. Then when you click **Refine** button, the command for refinement will output to your terminal window.

5. Predict

Click the **Prediction** tab at the upper most panel.

<img src="/Users/hengzhang/Library/Mobile Documents/com~apple~CloudDocs/projects/missingwedge/figures/gui_predick.png" alt="image-20220314193414624" style="zoom:33%;" />

Specify the GPU and choose the model for prediction. Click **Predict** to star predicting.

You can view the missing wedge corrected tomogram by clicking **3dmod**. 
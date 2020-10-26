#usr/bin/bash
mkdir ./tempdata
# mkdir ./masks
# mkdir ./noises
tomos=$@
for i in ${tomos[@]}
do 
    ln -s $i ./tempdata
    echo $i
done

#mwr_cli.py generate_noise ./noises 1000 --cubesize 64 
mwr_cli.py make_mask ./tempdata ./masks --mask_type "surface"
mwr_cli.py train --input_dir ./tempdata --mask_dir ./masks/ \
--noise_dir /storage/heng/mwrtest3D/noise64_2 \
--pretrained_model /storage/heng/mwrtest3D/pretrained_model/model_pp676.h5 \
--iterations 20   --ncube 300 --noise_level 0.07 \
--noise_start_iter 0 --noise_pause 5 --epochs 8 \
--batch_size 8  --gpuID 0,1,2,3 
rm -rf masks
rm -rf tempdata

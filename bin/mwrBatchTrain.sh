mkdir ./tempdata
mkdir ./masks
mkdir ./noises
tomos = "$*"
for i in "$tomos[@]"
do 
    ln -s ./tempdata $i
done

mwr_cli.py generate_noise ./noises 1000 --cubesize 64 
mwr_cli.py make_mask ./tempdata ./masks 
mwr_cli.py train --input-dir ./tempdata --gpuID '0,1,2,3' --noise_dir ./noises --pretrained_model neuron_pretrained.h5 --noise_dir noise64 --iterations 20 --steps_per_epoch 200 --nucbe 300 --noise_level 0.1 --noise_start_iter 0 --noise_pause 3 --epochs 10 --batch_size 8 

mkdir ./predict
tomos = "$*"
for i in "$tomos[@]"
do 
    mwr_cli.py predict $i $1
./results/model_iter40.h5
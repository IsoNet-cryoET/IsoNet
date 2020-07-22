# MWR

## Usage
![](turorials/images/origional_data.png)

### Training

Create a new folder and copy a setting template **settings.py** into it.

##### Configure Training

Training noise example: noise start to be added from iteration 20 and remain 0.04 for iteration 20-24, 0.08 for iteration 25-29 , so on. 

```
#level of noise STD(added noise)/STD(data)
noise_level = 0.04

#iteration to add trainning noise
noise_start_iter = 20

#iters trainning noise remain at one level
noise_pause = 5
```

**Note**: In current version, network configuring is invalid (like: *unet_depth* and *convs_per_depth* )

#### Run training

```
mwr3D settings.py
```

On DGX workstation

```
sudo docker run --runtime=nvidia  -t -v /localdata:/localdata -v /raid:/raid --rm tf-keras-mwr sh /raid/gqbi/missing_wedge/example/run.sh
```

#### Predicting

```
mwr3D_predict -h
```






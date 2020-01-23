'''
settings for unet2
'''
filter_base = 32
nconvs_per = 3
n_gpus = 4
batchnorm = True
dropout=0.5
last_activation = 'linear'
lr=0.0004
residual = True
loss = 'binary_crossentropy'
metrics = ['accuracy']
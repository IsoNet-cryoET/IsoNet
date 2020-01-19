import matplotlib.pyplot as plt
import numpy as np
import random
from tifffile import imsave
def present(x,y,n):
    imgx = np.squeeze(x,axis=-1)
    imgy = np.squeeze(y,axis=-1)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(imgx[n],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(imgy[n],cmap='gray')
    plt.show()
if __name__=='__main__':
    data = np.load('/home/heng/mwr/test_data/train_and_test_data2D.npz')
    (x,y,x_val,y_val) = (data['train_data'][0],data['train_data'][1],data['test_data'][0],data['test_data'][1])
    #present(x,y,1000)
    margin = 5
    img_width = 64
    img_height = 64
    n =6
    toshow=[]
    a=np.arange(0,23000,1)
    random.shuffle(a)
    for i in range(n*n):
        toshow.append(x[a[i]])
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin 
    stitched_filters = np.zeros((width, height, 1))
    
# fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img = toshow[i * n + j]
            width_margin = (img_width + margin) * i
            height_margin = (img_height + margin) * j
            stitched_filters[
                width_margin: width_margin + img_width,
                height_margin: height_margin + img_height, :] = img
    stitched_filters = stitched_filters.astype(np.uint8)
    imsave('sssssd.tif',stitched_filters)

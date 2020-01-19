from tifffile import imread,imsave
import numpy as py 
import prepare
data = imread('/home/heng/mwr/test_data/real_data/pp676-bin8-5i.tif')
data = np.expand_dims(data,axis=-1)
norm = prepare.PercentileNormalizer()
normalized = norm.before(data,'ZYXC')
restored ,scale= norm.after(normalized,None)
print('norm:',normalized)
print('restored',restored)

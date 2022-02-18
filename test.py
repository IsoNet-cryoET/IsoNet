#from scipy.spatial.transform.Rotation import random
import numpy as np
import scipy as sp
a=np.arange(1000).reshape(10,10,10)
print(a)

from scipy.stats import special_ortho_group
rot = special_ortho_group.rvs(3)

from scipy.ndimage import affine_transform
sh = (np.array(a.shape)-1)/2.
center = np.array(sh)
offset = center-np.dot(rot,center)
print(offset)
b=affine_transform(a,rot,offset=offset)
print(rot)
print(b)
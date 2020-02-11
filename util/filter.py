'''
from https://github.com/CSBDeep/CSBDeep/blob/master/csbdeep/data/generate.py
'''
import numpy as np
#from .image import norm_save
def no_background_patches(threshold=0.9, percentile=90):
    from scipy.ndimage.filters import maximum_filter, median_filter, gaussian_filter
    def _filter(datas, patch_size, dtype=np.float32):
        image = datas[0]
        if dtype is not None:
            image = image.astype(dtype)
        print('Gaussian_filter')
        image1 = gaussian_filter(image, 5)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        print('maximum_filter')
        filtered = maximum_filter(image1, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image1,percentile)
    return _filter

def no_background_patches_new(threshold=0.9, percentile=90):
    from scipy.ndimage.filters import maximum_filter, median_filter, gaussian_filter
    def _filter(image, patch_size, dtype=np.float32):
        if dtype is not None:
            image = image.astype(dtype)
        print('Gaussian_filter')
        image1 = gaussian_filter(image, 10)
        from mwr.util.image import norm_save
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        print('maximum_filter')
        filtered = maximum_filter(image1, patch_size, mode='constant')
        return filtered > threshold * np.percentile(image1,percentile)
    return _filter

def discard_slices(data,start,end):
    assert end-start<data.shape[0]
    return data[start:end]


import numpy as np

def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out


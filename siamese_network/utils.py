import torch
import numpy as np
from scipy.stats import multivariate_normal

def get_gaussian_mask(image_size=128, aspect_ratio=1/2): # aspect_ratio: 1:2
    #128 is image size
    # We will be using 256x128 patch instead of original 128x128 path because we are using for pedestrain with 1:2 AR.
    height = complex(0,image_size / aspect_ratio)
    width = complex(0, image_size)
    x, y = np.mgrid[0:1.0:height, 0:1.0:width] #128 is input size.
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5,0.5])
    sigma = np.array([0.22,0.22])
    covariance = np.diag(sigma**2) 
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
    z = z.reshape(x.shape) 

    z = z / z.max()
    z  = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask

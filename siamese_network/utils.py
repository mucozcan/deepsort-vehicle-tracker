import torch
import numpy as np
from scipy.stats import multivariate_normal


def get_gaussian_mask(height, width):  # aspect_ratio: 1:2

    height = complex(0, height)
    width = complex(0, width)
    x, y = np.mgrid[0:1.0:height, 0:1.0:width]  # 128 is input size.
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask

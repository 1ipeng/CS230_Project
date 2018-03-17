import numpy as np
from scipy.ndimage.filters import gaussian_filter

DIR_BINS = "./lab_result/100_train_lab/bins.npy"
lamda = 0.5
sigma = 5
Q = 313

bins = np.load(DIR_BINS).reshape(-1)
p = np.bincount(bins)
p = 1. * p / np.sum(p)


# Gaussian kernel missing
p = gaussian_filter(p, sigma)

w = 1. / ((1 - lamda) * p + 1. * lamda / Q)
w = w / np.sum(p * w)

print w
print np.sum(w * p)
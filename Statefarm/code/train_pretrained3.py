import os
import glob

import numpy as np
import pandas as pd
import pickle
import h5py
from scipy.misc import imread
from skimage import transform as sk_transform
from skimage import color as sk_color
from skimage import exposure
from skimage.restoration import denoise_bilateral
from keras.utils.generic_utils import Progbar

np.random.seed(16)
IMG_SHAPE = 224, 224

def _load_img(file, img_shape=(64, 64), grayscale=False):
    shape = list(img_shape) + [3]
    img = imread(file)
    # crop to right side
    #img = sk_transform.resize(img[:, -550:-70], shape, preserve_range=True)
    img = sk_transform.resize(img, shape, preserve_range=True)
    # swap axes to make (RGB, width, height) from (width, height, RGB)
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
	# Convert to BGR from RGB
    img = img[::-1, :, :].astype(np.float32)
    # normalize
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    for c in range(3):
        img[c, :, :] -= MEAN_VALUE[c]

path = os.path.join('../input/', '*.jpg')
files = sorted(glob.glob(path))

total = 0
for file in files:
    img = _load_img(file, img_shape, grayscale)
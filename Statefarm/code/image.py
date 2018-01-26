#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_otto import load_train_data
from scipy import ndimage as ndi
from skimage import feature, exposure
from skimage import io as sk_io
from skimage import transform as sk_transform
from skimage import color as sk_color
from skimage.restoration import nl_means_denoising
from skimage.restoration import denoise_bilateral

np.random.seed(4)
img_shape = 144, 144

def _load_img(file, img_shape=(64, 64), grayscale=False):
    shape = list(img_shape) + [3]

    img = sk_io.imread(file)
    assert img.shape == (480, 640, 3)

    # crop to right side
    img = sk_transform.resize(img[-470:-20, -550:-100], shape)

    if grayscale:
        img = sk_color.rgb2gray(img)
    return img

print("Loading images...")
path = os.path.join('../input', 'train', 'c9', '*.jpg')
files = glob.glob(path)

df = pd.read_csv(os.path.join('../input', 'driver_imgs_list.csv'))
dct_driver = {img_name: driver for img_name, driver in zip(df['img'], df['subject'])}

X, drivers = [], []

total = 0
for file in files:
    img = _load_img(file, img_shape, True)
    img_name = file.split(os.path.sep)[-1]
    driver = dct_driver[img_name]

    X.append(img)
    drivers.append(driver)
    total += 1
    if total%100 == 0:
		break

X = np.array(X).reshape(len(X), img_shape[0], img_shape[1], -1)
X = X.astype(np.float32)
drivers = np.array(drivers)

print(X.shape)

figs, axes = plt.subplots(5, 3, figsize=(10, 15))
[(ax.set_xticks([]), ax.set_yticks([]), ax.axis('off')) for ax in axes.flatten()]
for i, driver in enumerate(np.unique(drivers)[:5]):
    for j, n in enumerate(range(3)):
        #axes[i, j].imshow(denoise_bilateral(X[j*5+i,:,:,0], sigma_range=0.05, sigma_spatial=4), cmap='gray')
        axes[i, j].imshow(exposure.equalize_hist(X[j*5+i,:,:,0]), cmap='gray')
        axes[i, j].set_title(driver)
plt.show()
exit()

figs, axes = plt.subplots(5, 3, figsize=(10, 15))
[(ax.set_xticks([]), ax.set_yticks([]), ax.axis('off')) for ax in axes.flatten()]
for i, driver in enumerate(np.unique(drivers)[:5]):
    for j, n in enumerate(range(3)):
        axes[i, j].imshow(nl_means_denoising(X[j*5+i,:,:,:], 3, 4, 0.08))
        axes[i, j].set_title(driver)
plt.show()

exit()

# Compute the Canny filter for two values of sigma
edges1 = denoise_tv_chambolle(im, weight=0.1, multichannel=False)
ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('Original image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('Filtered' + str(j), fontsize=20)

fig.tight_layout()
plt.show()

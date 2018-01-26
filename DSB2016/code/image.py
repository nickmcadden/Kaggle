#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
from skimage.restoration import nl_means_denoising
from skimage.restoration import denoise_tv_chambolle

def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('/Users/nmcadden/kaggle/dsb2016/data/X_validate.npy')
    y = np.load('/Users/nmcadden/kaggle/dsb2016/data/y_train.npy')

    X = X.astype(np.float32)
    X /= 255

    return X, y

print('Loading training data...')
X, y = load_train_data()

#im = ndi.rotate(im, 15, mode='constant')
#im = ndi.gaussian_filter(im, 4)
#im += 0.2 * np.random.random(im.shape)

for j in range(1,130):
	for i in range(1,29):
		im = X[j, i+1] - X[j,i,:,:]
		if i == 1:
			im_ = im
		else:
			im_ = im_ + im

	im = np.sqrt(np.power(im,2))
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), sharex=True, sharey=True)

	#im = im[5:50,0:45]
	# display results

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

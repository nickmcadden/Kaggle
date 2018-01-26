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
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar

def _load_img(file, img_shape=(64, 64), grayscale=False):
    shape = list(img_shape) + [3]
    img = imread(file)
    # crop to right side
    #img = sk_transform.resize(img[:, -580:], shape, preserve_range=True)
    #img = sk_transform.resize(img[-486:-4, -576:], shape, preserve_range=True)
    img = sk_transform.resize(img[-485:, -585:], shape, preserve_range=True)
    #img = sk_transform.resize(img[-470:-20, -550:-30], shape, preserve_range=True)
    #img = sk_transform.resize(img[:, -550:-70], shape, preserve_range=True)
    #img = sk_transform.resize(img[:, -547:-67], shape, preserve_range=True)
    #img = sk_transform.resize(img, shape, preserve_range=True)
    # swap axes to make (RGB, width, height) from (width, height, RGB)
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
	# Convert to BGR from RGB
    img = img[::-1, :, :].astype(np.float32)
    # normalize
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    for c in range(3):
        img[c, :, :] -= MEAN_VALUE[c]

    return img

def nldenoise(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    print('Denoising images...')
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        X[i] = denoise_bilateral(X[i], sigma_range=0.05, sigma_spatial=4)
        progbar.add(1)

    return X

def equalize(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    print('Equalizing images...')
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        X[i] = exposure.equalize_hist(X[i])
        progbar.add(1)

    return X

def rotaterandom(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    print('Rotating images...')
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        if np.random.rand() > 0.5:
            angle = angle = (np.random.rand() - 0.5) * 12
            X[i] = rotate(X[i], angle, mode='nearest', reshape=False)
        progbar.add(1)

    return X
	
def load_train_data(
        main_path,
        img_shape=(64, 64),
        grayscale=False,
        max_img_per_class=None,
		zeromean=False,
		usecache=True):
    """Load training data
    main_path : path to train/test folder
    img_shape=(64, 64) : desired output image size
    swapaxes=True : if True, reshape to N x Color x Height x Width
    grayscale=False : if True, convert to grayscale
    max_img_per_class=None : if not None, only load first n images per class
    """

    X, y, drivers = [], [], []

    cache_path = os.path.join('..', 'input', 'cache', 'train_r_' + str(img_shape[0]) + '_c_' + str(img_shape[1]) + '_t_' + str(int(grayscale)) + '.dat')

    if usecache and os.path.isfile(cache_path):    
        print('Restoring training images from cache!')
        (X, y, drivers) = restore_data(cache_path)
    else:

        df = pd.read_csv(os.path.join(main_path, 'driver_imgs_list.csv'))
        dct_driver = {img_name: driver for img_name, driver in zip(df['img'], df['subject'])}

        X = np.zeros((22424, 3, img_shape[0], img_shape[1]))

        img_ix = 0
        for target in range(10):
            print('Load folder c{}'.format(target))
            path = os.path.join(main_path, 'train', 'c' + str(target), '*.jpg')
            files = glob.glob(path)
            if max_img_per_class:
                files = files[:max_img_per_class]

            for file in files:
                img = _load_img(file, img_shape, grayscale)

                img_name = file.split(os.path.sep)[-1]
                driver = dct_driver[img_name]

                X[img_ix,:,:,:] = img
                if img_ix%1000 == 0:
                    print(np.mean(X[img_ix,0,:,:]), np.mean(X[img_ix,1,:,:]), np.mean(X[img_ix,2,:,:]))
                img_ix +=1
                y.append(target)
                drivers.append(driver)

        X = X.astype(np.float32, copy=False)
        y = np.array(y).astype(np.int32)
        drivers = np.array(drivers)

        if usecache:
            cache_data([X, y, drivers], cache_path)

    #X = rotaterandom(X)

    return X, y, drivers


def load_test_data(
        main_path,
        img_shape=(64, 64),
        grayscale=False,
        img_start_ix=0,
        max_img=None,
        usecache=False):
    """Load test data
    main_path : path to train/test folder
    img_shape=(64, 64) : desired output image size
    swapaxes=True : if True, reshape to N x Color x Height x Width
    grayscale=False : if True, convert to grayscale
    return_ids=False : whether image names should be returned
    max_img=None : if not None, only load first n images
    """

    X, ids = [], []

    cache_path = os.path.join('..', 'input', 'cache', 'test_r_' + str(img_shape[0]) + '_c_' + str(img_shape[1]) + '_t_' + str(int(grayscale)) + '.dat')

    if usecache and os.path.isfile(cache_path):    
        print('Restoring test images from cache!')
        (X, ids) = restore_data(cache_path)

    else:
        path = os.path.join(main_path, 'test', '*.jpg')
        files = sorted(glob.glob(path))

        if max_img:
            files = files[img_start_ix:img_start_ix+max_img]

        total = 0
        for file in files:
            img = _load_img(file, img_shape, grayscale)
            X.append(img)
            img_name = file.split(os.path.sep)[-1]
            ids.append(img_name)
            total += 1
            if total%10000 == 0:
                print('Read {} images from {}'.format(total, len(files)))

        X = np.array(X).reshape(len(X), 3, img_shape[0], img_shape[1])
        X = X.astype(np.float32, copy=False)
        ids = np.array(ids)
        
        #if usecache:
            #cache_data([X, ids], cache_path)
    #X = rotaterandom(X)

    return X, ids


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        h5f = h5py.File(path, 'w')
        for i, obj in enumerate(data):
        	h5f.create_dataset('dataset'+str(i), data=obj, chunks=True)
        h5f.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        h5f = h5py.File(path,'r')
        data = h5f['dataset0'][:], h5f['dataset1'][:], h5f['dataset2'][:]
        h5f.close()
    return data


def make_submission(fname, y_proba, ids):
    """Make a submission file
    fname : name of file
    y_proba : class probabilities
    ids : image names
    """
    with open(fname, 'w') as f:
        f.write('img,' + ','.join('c' + str(i) for i in range(10)))
        f.write('\n')
        for row, id in zip(y_proba, ids):
            f.write(id + ',')
            f.write(','.join("{:.12f}".format(prob) for prob in row))
            f.write('\n')

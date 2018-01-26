import os
import glob
import numpy as np
import pandas as pd
import pickle, time
from scipy.misc import imread
from skimage import transform as sk_transform
from skimage import color as sk_color
from skimage import exposure
from skimage.restoration import denoise_bilateral
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar

def _load_img(file, img_shape=(224, 224), swapaxes=True, normalize=False):
    shape = list(img_shape) + [3]
    img = imread(file)
    img = sk_transform.resize(img, shape, preserve_range=True)
    if swapaxes:
        img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
        # Convert to BGR from RGB
        img = img[::-1, :, :].astype(np.float32)

    # deduct mean value for VGG
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    for c in range(3):
        img[c, :, :] -= MEAN_VALUE[c]

    if normalize:
        img /= 255

    return img

def flip180(img):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    img = img[:,::-1]

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

def flip180(img):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    img = img[:,::-1]

    return img

def load_train_data(img_shape=(224,224), rotate=False, display=False):
    X = []
    X_id = []
    y = []
    start_time = time.time()

    print('Loading training images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = _load_img(fl, img_shape)
            X.append(img)
            X_id.append(flbase)
            y.append(index)
            img = flip180(img)
            X.append(img)
            X_id.append(flbase)
            y.append(index)

    X = np.array(X).reshape(len(X), -1, img_shape[0], img_shape[1])
    X = X.astype(np.float32)
    y = np.array(y).astype(np.int32)

    if rotate == True:
        X = rotaterandom(X)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X, y, X_id

def load_test_data(img_start_ix=0, max_img=100, img_shape=(224,224)):
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    files = files[img_start_ix:img_start_ix+max_img]

    X = []
    X_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = _load_img(fl, img_shape)
        X.append(img)
        X_id.append(flbase)

    X = np.array(X).reshape(len(X), -1, img_shape[0], img_shape[1])
    X = X.astype(np.float32)

    #X = rotaterandom(X)

    return X, X_id

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
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT')
        f.write('\n')
        for row, id in zip(y_proba, ids):
            f.write(id + ',')
            f.write(','.join("{:.12f}".format(prob) for prob in row))
            f.write('\n')

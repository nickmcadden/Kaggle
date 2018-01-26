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

crop_coords = {'-1': [0,700,0,1000],
        '0':[250,700,400,1000], 
        '1':[250,700,400,1000], 
        '2':[200,600,150,720], 
        '3':[100,500,200,600], 
        '4':[150,650,350,900], 
        '5':[200,500,500,1000], 
        '6':[0,500,500,1000], 
        '7':[100,650,400,900], 
        '8':[0,600,0,600], 
        '9':[50,600,200,800],
        '10':[50,500,400,800],
        '11':[0,600,200,800],
        '12':[300,600,300,700],
        '13':[200,800,200,950],
        '14':[0,500,400,1000],
        '15':[0,300,200,800],
        '16':[200,1000,200,1000], 
        '17':[0,700,300,1000], 
        '18':[200,600,0,700], 
        '19':[250,700,400,1000],
        '20':[100,600,200,1000],
        '21':[100,600,0,700]}

def _load_img(file, img_shape=(224, 224), c=[0,1000,0,1000], swapaxes=True, normalize=False):
    shape = list(img_shape) + [3]
    img = imread(file)
    img = img[c[0]:c[1], c[2]:c[3]]
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

def load_train_data(img_shape=(224,224), rotate=False, display=False):
    X = []
    X_id = []
    y = []
    start_time = time.time()

    print('Loading training images')
    folder_to_target = {'ALB':0, 'BET':1, 'DOL':2, 'LAG':3, 'NoF':4, 'OTHER':5, 'SHARK':6, 'YFT':7}

    train_clusters = pd.read_csv('../output/train_clusters.csv')
    for index, row in train_clusters.iterrows():
        if index%500==0:
            print(index)
        fl = os.path.join('..', 'input', 'train', row['folder'], row['id'])
        img = _load_img(fl, img_shape, crop_coords[str(row['cluster'])])
        X.append(img)
        X_id.append(row['id'])
        y.append(folder_to_target[row['folder']])

    X = np.array(X).reshape(len(X), -1, img_shape[0], img_shape[1])
    X = X.astype(np.float32)
    y = np.array(y).astype(np.int32)

    if rotate == True:
        X = rotaterandom(X)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X, y, X_id

def load_pseudo_labels():
    # read a strong prediction file
    y = np.zeros(1000)
    preds = pd.read_csv(os.path.join('..', 'output', 'tnc_blend_zoomed_unzoomed.csv'))
    for index, row in preds.iterrows():
        num = np.random.rand()
        cum_total = 0
        for i in range(1,9):
            cum_total += row[i]
            if cum_total > num:
                y[index] = i-1
                #print(y[index], num, cum_total)
                break
    y = y.astype(np.int32)
    return y

def load_test_data(img_start_ix=0, max_img=100, img_shape=(224,224)):
    test_clusters = pd.read_csv('../output/test_clusters.csv')
    test_clusters = test_clusters[img_start_ix:img_start_ix+max_img]

    X = []
    X_id = []
    for index, row in test_clusters.iterrows():
        fl = os.path.join('..', 'input', 'test_stg1', row['id'])
        img = _load_img(fl, img_shape, crop_coords[str(row['cluster'])])
        X.append(img)
        X_id.append(row['id'])

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

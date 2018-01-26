# -*- coding: utf-8 -*-
import numpy as np
import os
import datetime
import pandas as pd
import model

from data import load_train_data
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from skimage import feature
from keras.regularizers import l2
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from scipy.misc import imread, imresize
from scipy import ndimage

# color type: 1 - grey, 3 - rgb
color_type_global = 1
np.random.seed(1)

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index


def preprocess(X):
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
           # X[i, j] = denoise_tv_chambolle(X[i, j], weight=0.1, multichannel=False)
		   X[i, j] = feature.canny(X[i, j], sigma=1.2)
        progbar.add(1)
    return X

def rotation_augmentation(X, angle_range):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_rot = np.copy(X)
    for i in range(len(X)):
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(X.shape[1]):
            X_rot[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1)
        progbar.add(1)
    return X_rot

# input image dimensions
img_rows, img_cols = 96, 96

X, y, drivers  = load_train_data(img_rows, img_cols, color_type_global)
y = np_utils.to_categorical(y, 10)
X, y, drivers  = shuffle(X, y, drivers, random_state=0)

unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',
                 'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',
                 'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',
                 'p081']

X_train, y_train, train_index = copy_selected_drivers(X, y, drivers, unique_list_train)
unique_list_valid = ['p075']
X_valid, y_valid, test_index = copy_selected_drivers(X, y, drivers, unique_list_valid)

cnn = model.get_model(img_rows, img_cols, color_type_global)
cnn.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid))



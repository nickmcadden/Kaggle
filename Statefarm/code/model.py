from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras import backend as K

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)

def get_model(img_rows, img_cols, color_type=1):
    model = Sequential()
    model.add(Convolution2D(16, 2, 2, border_mode='same', init='he_normal', input_shape=(color_type, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 2, 2, border_mode='same', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(96, 2, 2, border_mode='same', init='he_normal'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 2, 2, border_mode='same', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy', metrics=["accuracy"])
    return model

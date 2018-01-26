import numpy as np
import gc
import pickle
import theano
from lasagne import updates, nonlinearities, init
from lasagne import layers
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer
from lasagne.utils import floatX
from nolearn.lasagne import NeuralNet, BatchIterator
from sklearn.cross_validation import LabelKFold
from sklearn.utils import shuffle
from data_pretrained import load_train_data, load_test_data, make_submission
from utils_otto import CVTrainSplit, RotateBatchIterator, AdjustVariable

np.random.seed(15)
IMG_SHAPE = 224, 224
path = '../input'

X, y, drivers = load_train_data(path, grayscale=False, img_shape=IMG_SHAPE, equalize=False, zeromean=True, usecache=False)
X, y, drivers = shuffle(X, y, drivers, random_state=0)

train_split = CVTrainSplit(LabelKFold(drivers, n_folds=2))
batch_iterator_train = RotateBatchIterator(10, 0.5, 12, True)
batch_iterator_test = BatchIterator(12, False)

layer = [
    (InputLayer, {'shape': (None, 3, 224, 224)}),
    (Conv2DDNNLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DDNNLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DDNNLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 256, 'filter_size': 3, 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (Conv2DDNNLayer, {'num_filters': 512, 'filter_size': 3, 'pad': 1}),
    (MaxPool2DLayer, {'pool_size': 2}),
    (DenseLayer, {'num_units': 4096}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 4096}),
    (DropoutLayer, {'p': 0.5}),
    (DenseLayer, {'num_units': 10, 'nonlinearity': nonlinearities.softmax})
]

net = NeuralNet(
    layers=layer, 
    update=updates.nesterov_momentum,
    update_momentum=0.9,
    update_learning_rate=theano.shared(floatX(0.001)),
	batch_iterator_train=batch_iterator_train,
	batch_iterator_test=batch_iterator_test,
    verbose=1,
    train_split=train_split,
    max_epochs=4,
)

# load pretrained model
with open('../input/pretrained/vgg16.pkl', 'rb') as f:
    params = pickle.load(f)

# replace last 2 param layers ((4096,1000)) and (1000,) with ((4096,10)) and (10,)
params['param values'][30] = params['param values'][30][:,:10]
params['param values'][31] = params['param values'][31][:10]

net.initialize_layers()
layers.set_all_param_values(net.layers_.values(), params['param values'])

print("Training neural network...")
net.fit(X, y)

del X
X_test, ids = load_test_data(path, grayscale=False, img_shape=IMG_SHAPE)

print("Predicting on test data...")
y_proba = net.predict_proba(X_test)
make_submission('../output/submission_01.csv', y_proba, ids)
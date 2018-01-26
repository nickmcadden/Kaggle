import numpy as np
import gc
import theano
from lasagne import layers, updates, nonlinearities, init
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.visualize import plot_occlusion, plot_loss
from sklearn.cross_validation import LabelKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from data_otto import load_train_data, load_test_data, make_submission
from utils_otto import CVTrainSplit, RotateBatchIterator, AdjustVariable

def float32(k):
	return np.cast['float32'](k)

gc.collect()
np.random.seed(12)
IMG_SHAPE = 96, 96
path = '../input'

X, y, drivers = load_train_data(path, grayscale=False, img_shape=IMG_SHAPE, equalize=False, zeromean=True)
X, y, drivers = shuffle(X, y, drivers, random_state=0)

print("input shape", X.shape)

train_split = CVTrainSplit(LabelKFold(drivers, n_folds=5))
batch_iterator_train = RotateBatchIterator(10, 0.5, 128, True)

layer = [
    (layers.InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),
    (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': 3, 'pad': 'same', 'nonlinearity': nonlinearities.rectify}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.Conv2DLayer, {'num_filters': 64, 'filter_size': 3, 'pad': 'same', 'nonlinearity': nonlinearities.rectify}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.Conv2DLayer, {'num_filters': 128, 'filter_size': 3, 'pad': 'same', 'nonlinearity': nonlinearities.rectify}),
    (layers.MaxPool2DLayer, {'pool_size': 8}),
    (layers.DropoutLayer, {'p': 0.5}),
    (layers.DenseLayer, {'num_units': 10, 'nonlinearity': nonlinearities.softmax})
]

net = NeuralNet(
    layers=layer,
    update=updates.adam,
    update_learning_rate=theano.shared(float32(0.001)),
    verbose=1,
    batch_iterator_train=batch_iterator_train,
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.001, stop=0.0002)],
    eval_size=0.0,
    max_epochs=10,
)

print("Training neural network...")
gc.collect()
net.fit(X, y)

X_test, ids = load_test_data(path, grayscale=False, img_shape=IMG_SHAPE, equalize=False, zeromean=True)

print("Predicting on test data...")
y_proba = net.predict_proba(X_test)
make_submission('../output/submission_10.csv', y_proba, ids)

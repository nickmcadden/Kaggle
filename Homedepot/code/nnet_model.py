import numpy as np
import pandas as pd
import theano

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
from lasagne.layers import DenseLayer, InputLayer, DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad, adam
from nolearn.lasagne import NeuralNet
from ml_metrics import quadratic_weighted_kappa
import theano.tensor as T

def root_mean_square_error(y_true, y_pred):
    epsilon = 10e-9
    return T.sqrt(((y_pred - y_true) ** 2).mean(axis=-1))

class EarlyStopping(object):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdjustVariable(object):
    def __init__(self, name, start=0.0002, stop=0.0002):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def float32(k):
	return np.cast['float32'](k)


def nnet(d0,h1,d1,h2,d2,e,l,num_features, runtype):			
	if runtype is 'cv':
		on_epoch_finished=[
	        AdjustVariable('update_learning_rate', start=l, stop=l/10),
			EarlyStopping(patience=15)
	        ]
	elif runtype is 'actual':
		on_epoch_finished=[
		    AdjustVariable('update_learning_rate', start=l, stop=l/10)
		    ]
	layers0 = [
			('input', InputLayer),
			('dropout0', DropoutLayer),
			('hidden1', DenseLayer),
			('dropout1', DropoutLayer),
			('hidden2', DenseLayer),
			('dropout2', DropoutLayer),
			#('hidden3', DenseLayer),
			#('dropout3', DropoutLayer),
			('output', DenseLayer)
			]

	net0 = NeuralNet(
			layers=layers0,
			input_shape=(None, num_features),
			dropout0_p=d0,
			hidden1_num_units=h1,
			dropout1_p=d1,
			hidden2_num_units=h2,
			dropout2_p=d2,
			output_num_units=1,
			output_nonlinearity=None,
			objective_loss_function=root_mean_square_error,
			update=adam,
			update_learning_rate=theano.shared(float32(l)),
			#update_momentum=0.9,
			regression=True,
			verbose=1,
			max_epochs=e)
	return net0

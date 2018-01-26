import numpy as np
import pandas as pd
import theano
import sys
import copy
import data as data
import argparse

from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano.tensor as T
from sklearn.metrics import mean_squared_error

print(theano.config.floatX)

def root_mean_square_log_error(y_true, y_pred):
    epsilon = 10e-9
    return T.sqrt(((T.log(T.clip(y_pred, epsilon, np.inf) + 1.) - T.log(T.clip(y_true, epsilon, np.inf) + 1.)) ** 2).mean(axis=-1))

def np_root_mean_square_log_error(y_true, y_pred):
    epsilon = 10e-9
    return np.sqrt(((np.log1p(np.clip(y_pred, epsilon, np.inf) + 1.) - np.log1p(np.clip(y_true, epsilon, np.inf) + 1.)) ** 2).mean(axis=-1))

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=100)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=50)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='tgtrate')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

def log_loss(act, pred):
    """ Vectorised computation of logloss """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

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
    def __init__(self, name, start=0.005, stop=0.002):
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

def make_submission(clf, X_test, ids, name):
	name = "../output/" + name + ".csv"
	pred = clf.predict(X_test)
	pred = np.clip(pred, 0, 9999)
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": pred[:,0]})
	print(preds.shape, np.min(preds['Demanda_uni_equil']))
	preds.to_csv(name, float_format='%.3f', index=False)
	print("Wrote submission to file {}.".format(name))

def init_nnet(d0,h1,d1,h2,d2,e,l,runtype):
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
    update=adam,
    update_learning_rate=theano.shared(np.float32(0.001)),
	output_nonlinearity=None,
	objective_loss_function=root_mean_square_log_error,
    regression=True,
    y_tensor_type = T.imatrix,                   
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=e,
    eval_size=l,
    verbose=2)
	return net0

# Get the data
np.random.seed(2)
# Load data
X, y, X_test, ids = data.load(m_params)

val_ix = (X[:,0] > 7)
tr_ix = (X[:,0] <= 7)

X = X.astype(theano.config.floatX)
y = y.astype(np.int32)
X_test = X_test.astype(theano.config.floatX)
num_features = X.shape[1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# initialise variables for main loop
d0, d1, d2, h1, h2 = 0.05, 0.05, 0.05, 500, 500
e = 5
l = 0.2

if m_params['cv']:
	clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')	
	print(X[tr_ix].shape, X[val_ix].shape)
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict(X[val_ix])
	scr = np.sqrt(mean_squared_error(np.log1p(np.clip(y[val_ix], 0, 99999)), np.log1p(np.clip(pred, 0, 99999))))
	print('Train score is:', scr)

else:
	clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')
	clf.eval_size=0
	# fit the model
	clf.fit(X, y)
	#submission filename
	subname = "d" + str(int(d0*100)) + "_h" + str(h1) + "_d" + str(int(d1*100)) + "_h" + str(h2) + "_d" + str(int(d2*100)) + "_e" + str(e) + "_l" + str(l)
	subname = "nnet_" + "_" + subname
	print("Saving Results.")
	make_submission(clf, X_test, ids, subname)

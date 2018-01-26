import numpy as np
import pandas as pd
import sys
import copy
import data_rejigged as data
import argparse

from pprint import pprint
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.objectives import binary_crossentropy
from lasagne.nonlinearities import sigmoid
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet
import theano
import theano.tensor as T

parser = argparse.ArgumentParser(description='Option parser')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=20)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=1)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
m_params = vars(parser.parse_args())

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
    def __init__(self, name, start=0.02, stop=0.005):
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
	name = "output/" + name + ".csv"
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids)
	preds.to_csv(name, index_label='id', float_format='%.3f')
	print("Wrote submission to file {}.".format(name))

def init_nnet(d0,h1,d1,h2,d2,e,l,runtype):			
	if runtype is 'cv':
		on_epoch_finished=[
	        AdjustVariable('update_learning_rate', start=l, stop=l/3),
			EarlyStopping(patience=15)
	        ]
	elif runtype is 'actual':
		on_epoch_finished=[
		    AdjustVariable('update_learning_rate', start=l, stop=l/3)
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
			hidden1_nonlinearity=sigmoid, 
			dropout1_p=d1,
			hidden2_num_units=h2,
			dropout2_p=d2,
			output_num_units=1,
			update=adagrad,
			output_nonlinearity=sigmoid, 
			update_learning_rate=theano.shared(float32(l)),
			objective_loss_function = binary_crossentropy,
			#update_momentum=0.9,
			regression=True,
			#y_tensor_type = T.imatrix,
			verbose=1,
			max_epochs=e)
	return net0

def eval_nnet(d0,h1,d1,h2,d2,e,l):
	netcv = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')
	subname = "d" + str(int(d0*100)) + "_h" + str(h1) + "_d" + str(int(d1*100)) + "_h" + str(h2) + "_d" + str(int(d2*100)) + "_e" + str(e) + "_l" + str(l)
	print(subname)
	# fit the model
	netcv.fit(X, y)
	# add score to submission filename
	cvscore = netcv.train_history_[-1]
	make_submission(netcv, X_test, ids, subname)
	if cvscore['valid_loss'] < 0.46:
		pprint(cvscore)
		net1 = init_nnet(d0,h1,d1,h2,d2,e,l,'actual')
		net1.eval_size=0
		# fit the model
		net1.fit(X, y)
		# add score to submission filename
		subname = "nnet_" + "{:.4f}".format(cvscore['valid_loss']) + "_" + subname
		make_submission(net1, X_test, ids, subname)
	return (cvscore['train_loss']/cvscore['valid_loss']), cvscore['valid_loss']

# Get the data
np.random.seed(1)
# Load data
X, y, X_test, ids = data.load(m_params)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(theano.config.floatX)
y = y.reshape(-1, 1).astype(np.float32)
X_test = scaler.transform(X_test).astype(theano.config.floatX)
num_features = X.shape[1]

# initialise variables for main loop
i, max_iter = 1 , 1
d0_ctr, d1_ctr, d2_ctr, h1_ctr, h2_ctr = 0.4, 0.25, 0.05, 900, 256
e, l = 10, 0.015
bst_scr = 1

while i <= max_iter:
	d0 = d0_ctr
	d1 = np.random.normal(d1_ctr, d1_ctr/12)
	d2 = np.random.normal(d2_ctr, d2_ctr/7)
	h1 = int(np.random.normal(h1_ctr, h1_ctr/20))
	h2 = int(np.random.normal(h2_ctr, h2_ctr/12))
	tr_val_ratio, vl_scr = eval_nnet(d0,h1,d1,h2,d2,e,l)
	if vl_scr < bst_scr and tr_val_ratio > 0.94:
		bst_scr = vl_scr
		d0_ctr = (d0 + d0_ctr) / 2
		d1_ctr = (d1 + d1_ctr) / 2
		d2_ctr = (d2 + d2_ctr) / 2
		h1_ctr = (h1 + h1_ctr) / 2
		h2_ctr = (h2 + h2_ctr) / 2
	print(bst_scr)
	print(d0_ctr,d1_ctr,d2_ctr,h1_ctr,h2_ctr)
	X, y, X_test, ids = data.load(m_params)
	i += 1

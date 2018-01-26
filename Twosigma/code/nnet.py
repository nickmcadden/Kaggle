import numpy as np
import pandas as pd
import theano
import sys
import copy
import data_nnet as data
import argparse
import pickle as pkl

from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.metrics import log_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, tanh, leaky_rectify as rectify
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet

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

def make_submission(clf, X_sub, ids, name):
	name = "../output/" + name + ".csv"
	y_prob = clf.predict_proba(X_sub)
	preds = pd.DataFrame(y_prob, index=ids)
	preds.to_csv(name, index_label='id', float_format='%.3f')
	print("Wrote submission to file {}.".format(name))

def init_nnet(d0,h1,d1,h2,d2,h3,d3,e,l,runtype):
	layers0 = [
			('input', InputLayer),
			('dropout0', DropoutLayer),
			('hidden1', DenseLayer),
			('dropout1', DropoutLayer),
			('hidden2', DenseLayer),
			('dropout2', DropoutLayer),
			('hidden3', DenseLayer),
			('dropout3', DropoutLayer),
			('output', DenseLayer)
			]

	net0 = NeuralNet(
		layers=layers0,
		input_shape=(None, num_features),
		dropout0_p=d0,
		hidden1_num_units=h1,
		hidden1_nonlinearity=rectify,
		dropout1_p=d1,
		hidden2_num_units=h2,
		hidden2_nonlinearity=rectify,
		dropout2_p=d2,
		hidden3_num_units=h3,
		hidden3_nonlinearity=rectify,
		dropout3_p=d3,
		output_num_units=3,
		output_nonlinearity=softmax,
		update=adagrad,
		update_learning_rate=theano.shared(float32(l)),
		#on_epoch_finished=on_epoch_finished,
		#update_momentum=0.9,
		eval_size=0.2,
		max_epochs=e,
		verbose=2)
	return net0

# Get the data
np.random.seed(2)
# Load data
X, y, X_sub, ids = data.load(m_params)

X = X.astype(theano.config.floatX)
y = y.astype(np.int32)
X_sub = X_sub.astype(theano.config.floatX)
num_features = X.shape[1]

# initialise variables for main loop
d0, d1, d2, d3, h1, h2, h3 = 0.3, 0.2, 0.2, 0.2, 500, 200, 100
e = 100
l = 0.01

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))
for i, (tr_ix, val_ix) in enumerate(kf):
	clf = init_nnet(d0,h1,d1,h2,d2,h3,d3,e,l,'cv')	
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict_proba(X[val_ix])
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict_proba(X_sub) / 5
	scr[i] = log_loss(y[val_ix], np.array(pred))
	print('Train score is:', scr[i])
print(log_loss(y, oob_pred))
oob_pred_filename = '../output/oob_nnet_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_nnet_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/nnet' + str(np.mean(scr)) + '.csv', index=False)

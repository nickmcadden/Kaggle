import numpy as np
import pandas as pd
import theano
import sys
import copy
import data_nnet as data
import argparse
import scipy as sp
import pickle as pkl

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
from sklearn.metrics import roc_auc_score

print(theano.config.floatX)

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
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids)
	preds.to_csv(name, index_label='id', float_format='%.3f')
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
    output_nonlinearity=sigmoid,
    update=adagrad,
    update_learning_rate=theano.shared(np.float32(0.01)),
    #update_momentum=theano.shared(np.float32(0.9)),    
    #Decay the learning rate
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.001, stop=0.001),
                       #AdjustVariable('update_momentum', start=0.9, stop=0.99),
                       ],
    regression=True,
    y_tensor_type = T.imatrix,                   
    objective_loss_function = binary_crossentropy,
    #batch_iterator_train = BatchIterator(batch_size = 256),
    max_epochs=e,
    eval_size=l,
    verbose=2)
	return net0

# Get the data
np.random.seed(2)
# Load data
X, y, X_test, ids, zeromap = data.load(m_params)
X = X.astype(theano.config.floatX)
y = y.astype(np.int32)
X_test = X_test.astype(theano.config.floatX)
num_features = X.shape[1]

# initialise variables for main loop
d0, d1, d2, h1, h2 = 0.05, 0.2, 0.1, 1200, 1200
e = 40
l = 0.2

if m_params['cv']:
	# do cross validation scoring
	kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_test.shape[0], 5))
	for i, (tr_ix, val_ix) in enumerate(kf):
		clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')	
		clf.fit(X[tr_ix], y[tr_ix])
		pred = clf.predict_proba(X[val_ix])
		oob_pred[val_ix] = np.array(pred)
		sub_pred[:,i] = np.squeeze(clf.predict_proba(X_test))
		scr[i] = roc_auc_score(y[val_ix], oob_pred[val_ix])
		print('Train score is:', scr[i])
	print np.mean(scr)
	print sub_pred[1:10]
	sub_pred = sub_pred.mean(axis=1) * zeromap
	oob_pred_filename = '../output/oob_nnet_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_nnet_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"ID": ids, "target": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:
	clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')
	clf.eval_size=0
	# fit the model
	clf.fit(X, y)
	#submission filename
	subname = "d" + str(int(d0*100)) + "_h" + str(h1) + "_d" + str(int(d1*100)) + "_h" + str(h2) + "_d" + str(int(d2*100)) + "_e" + str(e) + "_l" + str(l)
	subname = "nnet_" + "{:.4f}".format(cvscore['valid_loss']) + "_" + subname
	print("Saving Results.")
	make_submission(clf, X_test, ids, subname)

import numpy as np
import pandas as pd
import theano
import sys
import copy
import data3 as data
import argparse
import scipy as sp
from scipy import sparse
import pickle as pkl

from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.metrics import log_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet, TrainSplit
import theano.tensor as T

print(theano.config.floatX)

parser = argparse.ArgumentParser(description='NNet for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of iterations', type=int, default=100)
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

def make_submission(clf, X_test, ids, name):
	name = "../output/" + name + ".csv"
	pred = clf.predict_proba(X_test)
	preds = pd.DataFrame({"listing_id": ids, "high": pred[:,0], "medium": pred[:,1], "low": pred[:,2]})
	preds = preds[["listing_id", "high", "medium", "low"]]
	preds.to_csv('../output/nnet_blend.csv', index=False)
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
		hidden1_nonlinearity=tanh,
		dropout1_p=d1,
		hidden2_num_units=h2,
		hidden2_nonlinearity=sigmoid,
		dropout2_p=d2,
		hidden3_num_units=h3,
		hidden3_nonlinearity=sigmoid,
		dropout3_p=d3,
		output_num_units=3,
		output_nonlinearity=softmax,
		update=adagrad,
		update_learning_rate=theano.shared(float32(l)),
		#on_epoch_finished=on_epoch_finished,
		#update_momentum=0.9,
		train_split=TrainSplit(eval_size=0.2),
		max_epochs=e,
		verbose=2)
	return net0

# Get the data
np.random.seed(2)
# Load data
X, y, X_sub, ids = data.load(m_params)

X=X[:,:5]
X_sub=X_sub[:,:5]

print("Loading OOB predictions...\n") 

oob_models = [	'oob_pred_rfentropy_0.558171910933.p',
				'oob_pred_xgb_reg_0.210871555251.p',
				'oob_pred_xgb_reg_0.213364691979.p',
				'oob_pred_gbr_reg_0.228964054965.p',
				'oob_pred_etcentropy_0.551557316458.p',
				'oob_pred_adaboost_0.592650009197.p',
				#'oob_nnet_0.574446898612.p',
				#'oob_pred_linreg_0.245916676855.p',
				'oob_pred_lr_0.574865487204.p',
				'oob_pred_gbcentropy_0.550568716867.p',
				'oob_pred_gbcentropy_0.56107032581.p',
				'oob_pred_xgb_0.522546789983.p',
				'oob_pred_xgb_0.524605390397.p',
				'oob_pred_xgb_0.524458909366.p',
				'oob_pred_xgb_0.525800613352.p',
				'oob_pred_xgb_0.529575127857.p',
				'oob_pred_xgb_0.52390324817.p']

for oob_model_name in oob_models:
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))
	oob_model = (oob_model-np.mean(oob_model))/np.std(oob_model)
	sub_model = (sub_model-np.mean(sub_model))/np.std(sub_model)
	X = np.hstack([X, oob_model])
	X_sub = np.hstack([X_sub, sub_model])

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

X = X.astype(theano.config.floatX)
y = y.astype(np.int32)
X_sub = X_sub.astype(theano.config.floatX)
num_features = X.shape[1]

# initialise variables for main loop
d0, d1, d2, d3, h1, h2, h3 = 0, 0.1, 0.1, 0.1, 60, 45, 25
e = 150
l = 0.02

if m_params['cv']:
	# do cross validation scoring
	kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros((X.shape[0], 3))
	for i, (tr_ix, val_ix) in enumerate(kf):
		clf = init_nnet(d0,h1,d1,h2,d2,h3,d3,e,l,'cv')	
		clf.fit(X[tr_ix], y[tr_ix])
		pred = clf.predict_proba(X[val_ix])
		oob_pred[val_ix] = np.array(pred)
		scr[i] = log_loss(y[val_ix], oob_pred[val_ix])
		print('Train score is:', scr[i])
	print(log_loss(y, oob_pred))
	print oob_pred[1:10]
	oob_filename = '../output/oob_blend_nnet' + str(np.mean(scr)) + '.p'
	pkl.dump(oob_pred, open(oob_filename, 'wb'))

else:
	clf = init_nnet(d0,h1,d1,h2,d2,h3,d3,e,l,'cv')
	clf.train_split=TrainSplit(eval_size=0)
	# fit the model
	clf.fit(X, y)
	#submission filename
	subname = "nnet_blend"
	print("Saving Results.")
	make_submission(clf, X_sub, ids, subname)

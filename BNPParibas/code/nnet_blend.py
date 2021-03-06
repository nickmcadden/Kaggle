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
from sklearn.cross_validation import StratifiedShuffleSplit, KFold

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, tanh
from lasagne.updates import nesterov_momentum, adagrad, adam
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
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
    on_epoch_finished=[AdjustVariable('update_learning_rate', start=0.002, stop=0.001),
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
X, y, X_sub, ids = data.load(m_params)

print("Loading OOB predictions...\n") 

oob_models = [	'oob_pred_etcentropy_0.456815287264.p',
				'oob_pred_etcgini_0.458635685313.p',
				'oob_pred_etclinearfeatures_0.455779519257.p',
				'oob_pred_xgb_0.461258027149.p',
				'oob_pred_xgb_0.460586010411.p',
				'oob_pred_gbcdeviance_0.460854641372.p',
				'oob_nnet_0.467769825293.p',
				'oob_pred_logit_0.482244107768.p',
				'oob_pred_rfentropy_0.464300125336.p']

for oob_model_name in oob_models:
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))	
	X = np.column_stack((X, (oob_model-np.mean(oob_model))/np.std(oob_model)))
	X_sub = np.column_stack((X_sub, (sub_model-np.mean(sub_model))/np.std(sub_model)))
	
X = X.astype(theano.config.floatX)
y = y.astype(np.int32)
X_sub = X_sub.astype(theano.config.floatX)
num_features = X.shape[1]

# initialise variables for main loop
d0, d1, d2, h1, h2 = 0.05, 0.2, 0.1, 3000, 1000
e = 35
l = 0.1

if m_params['cv']:
	# do cross validation scoring
	kf = KFold(X.shape[0], n_folds=4, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	for i, (tr_ix, val_ix) in enumerate(kf):
		clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')	
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
	clf = init_nnet(d0,h1,d1,h2,d2,e,l,'cv')
	clf.eval_size=0
	# fit the model
	clf.fit(X, y)
	#submission filename
	subname = "pred_nnet_blend"
	print("Saving Results.")
	make_submission(clf, X_sub, ids, subname)

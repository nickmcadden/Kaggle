import numpy as np
import pandas as pd
import sys
import copy

from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
	df = pd.read_csv(path)
	X = df.values.copy()
	X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(np.int32)
	X = scaler.transform(X)
	return X, ids
	
def make_submission(clf, X_test, ids, encoder, name):
	name = "lasagne/" + name + ".csv"
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.3f')
	print("Wrote submission to file {}.".format(name))

def init_nnet(d0,h1,d1,h2,d2,e,l):			
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
			#hidden3_num_units=128,
			#dropout3_p=0.2,
			output_num_units=num_classes,
			output_nonlinearity=softmax,
			update=adagrad,
			update_learning_rate=l,
			#update_momentum=0.9,
			eval_size=0.2,
			verbose=1,
			max_epochs=e)
			
	return net0

def eval_nnet(d0,h1,d1,h2,d2,e,l):
	netcv = init_nnet(d0,h1,d1,h2,d2,e,l)
	subname = "d" + str(int(d0*100)) + "_h" + str(h1) + "_d" + str(int(d1*100)) + "_h" + str(h2) + "_d" + str(int(d2*100)) + "_e" + str(e) + "_l" + str(l)
	print(subname)
	# fit the model
	netcv.fit(X, y)
	# add score to submission filename
	cvscore = netcv.train_history_[e-1]
	if cvscore['valid_loss'] < 0.456 and (cvscore['train_loss']/cvscore['valid_loss'] > 0):
		pprint(cvscore)
		net1 = init_nnet(d0,h1,d1,h2,d2,e,l)
		net1.eval_size=0
		# fit the model
		net1.fit(X, y)
		# add score to submission filename
		subname = "nnet_" + "{:.4f}".format(cvscore['valid_loss']) + "_" + subname
		make_submission(net1, X_test, ids, encoder, subname)
	return cvscore['train_loss'], cvscore['valid_loss']

# Get the data
np.random.seed(2)
X, y, encoder, scaler = load_train_data('trainlg.csv')
X_test, ids = load_test_data('testlg.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# initialise variables for main loop
i, max_iter = 1 , 3
d0_ctr, d1_ctr, d2_ctr, h1_ctr, h2_ctr = 0.17, 0.25, 0.22, 960, 530
e, l = 200, 0.005
bst_scr = 0.46

while i <= max_iter:
	X, y, encoder, scaler = load_train_data('trainlg.csv')
	d0 = d0_ctr
	d1 = np.random.normal(d1_ctr, d1_ctr/12)
	d2 = np.random.normal(d2_ctr, d2_ctr/7)
	h1 = int(np.random.normal(h1_ctr, h1_ctr/20))
	h2 = int(np.random.normal(h2_ctr, h2_ctr/12))
	tr_scr, vl_scr = eval_nnet(d0,h1,d1,h2,d2,e,l)
	if vl_scr < bst_scr:
		bst_scr = vl_scr
		d0_ctr = (d0 + d0_ctr) / 2
		d1_ctr = (d1 + d1_ctr) / 2
		d2_ctr = (d2 + d2_ctr) / 2
		h1_ctr = (h1 + h1_ctr) / 2
		h2_ctr = (h2 + h2_ctr) / 2
	print(bst_scr)
	print(d0_ctr,d1_ctr,d2_ctr,h1_ctr,h2_ctr)	
	i += 1

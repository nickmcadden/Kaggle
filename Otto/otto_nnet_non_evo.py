import numpy as np
import pandas as pd
import theano
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

class EarlyStopping(object):
    def __init__(self, patience=100):
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

def init_nnet(d0,h1,d1,h2,d2,e,l,runtype):			
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
			#hidden3_num_units=128,
			#dropout3_p=0.2,
			output_num_units=num_classes,
			output_nonlinearity=softmax,
			update=adagrad,
			update_learning_rate=theano.shared(float32(l)),
			on_epoch_finished=on_epoch_finished,
			#update_momentum=0.9,
			eval_size=0.2,
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
	if cvscore['valid_loss'] < 0.456:
		pprint(cvscore)
		net1 = init_nnet(d0,h1,d1,h2,d2,e,l,'actual')
		net1.eval_size=0
		# fit the model
		net1.fit(X, y)
		# add score to submission filename
		subname = "nnet_" + "{:.4f}".format(cvscore['valid_loss']) + "_" + subname
		make_submission(net1, X_test, ids, encoder, subname)
	return (cvscore['train_loss']/cvscore['valid_loss']), cvscore['valid_loss']

# Get the data
np.random.seed(5)
X, y, encoder, scaler = load_train_data('trainlg.csv')
X_test, ids = load_test_data('testlg.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

# initialise variables for main loop
i, max_iter = 1 , 20
d0, d1, d2, h1, h2 = 0.17, 0.27, 0.31, 981, 583
e, l = 175, 0.01
bst_scr = 0.46

while i <= max_iter:
	tr_val_ratio, vl_scr = eval_nnet(d0,h1,d1,h2,d2,e,l)
	if vl_scr < bst_scr and tr_val_ratio > 0.94:
		bst_scr = vl_scr
	print(bst_scr)
	X, y, encoder, scaler = load_train_data('trainlg.csv')	
	i += 1

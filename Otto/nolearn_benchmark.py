import numpy as np
import pandas as pd
import sys

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
			
X, y, encoder, scaler = load_train_data('trainlg.csv')
X_test, ids = load_test_data('testlg.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]


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
		dropout0_p=0.15,
		hidden1_num_units=1000,
		dropout1_p=0.25,
		hidden2_num_units=500,
		dropout2_p=0.25,
		#hidden3_num_units=128,
		#dropout3_p=0.2,
		output_num_units=num_classes,
		output_nonlinearity=softmax,
		update=adagrad,
		update_learning_rate=0.01,
		#update_momentum=0.9,
		eval_size=0.2,
		verbose=1,
		max_epochs=150)

d = net0.__dict__
e = d['max_epochs']
subname = "d" + str(int(d['dropout0_p']*100)) + "_h" + str(d['hidden1_num_units']) + "_d" + str(int(d['dropout1_p']*100)) + "_h" + str(d['hidden2_num_units']) + "_d" + str(int(d['dropout2_p']*100)) + "_e" + str(d['max_epochs']) + "_l" + str(d['update_learning_rate'])
print(subname)
# fit the model
net0.fit(X, y)

net0.eval_size=0
# fit the model
net0.fit(X, y)

# add score to submission filename
score = "{:.4f}".format(net0.train_history_[e-1]['valid_loss'])
subname = "nnet_" + score + "_" + subname
make_submission(net0, X_test, ids, encoder, subname)

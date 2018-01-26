import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data_testing as data
import argparse
import pickle as pkl
from time import time

from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def log_loss(act, pred):
    """ Vectorised computation of logloss """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='std')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("TSNE...\n")
t1 = time()
model = TSNE(n_components=2, verbose=1)
print(model)
tsne_vec = np.zeros((1000, 2))
for a in [1000]:
	tsne_part_model = model.fit_transform(X[a-1000:a, :],y)
	tsne_vec[a-1000:a, :] = tsne_part_model

print(tsne_vec[0:30])
print("time", time() - t1)
print("done")

plt.scatter(list(tsne_vec[:,0]), list(tsne_vec[:,1]), c=list(y[0:1000]), cmap='summer')
plt.show()


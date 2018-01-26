import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data
import pickle as pkl
import argparse
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
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
    ll = sum(   act*sp.log(pred) + 
                sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

print("reading the train data\n")
X = pd.read_csv('../input/train.csv')
y = X['target']

model1 = pkl.load(open('../output/oob_blend_nnet0.449510656577.p','rb'))
model2 = pkl.load(open('../output/oob_blend_ens0.449199269125.p', 'rb'))
model3 = pkl.load(open('../output/oob_pred_xgb_0.461258027149.p', 'rb'))

print model1[1:10]
print model2[1:10]
print y[1:10]

oob_pred = np.zeros((X.shape[0], 3))
oob_pred[:,0] = model1
oob_pred[:,1] = model2
oob_pred[:,2] = model3

print("Blending.")
blend = LinearRegression(fit_intercept=False)
blend.fit(oob_pred, y)
print("coefs",blend.coef_)

print('model 1', log_loss(y, model1))
print('model 2', log_loss(y, model2))
print('model 3', log_loss(y, model3))
print('model avg 1 2', log_loss(y, (model1*0.55+model2*0.45)))
print('model weighted', log_loss(y, (model1*0.7+model2*0.15+model3*0.15)))
print('model LR', log_loss(y, (model1*blend.coef_[0]+model2*blend.coef_[1]+model3*blend.coef_[2])))

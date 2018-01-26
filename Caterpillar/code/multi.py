import time, pickle
import pandas as pd
import numpy as np
import data
import xgboost as xgb
from sklearn import ensemble, preprocessing
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
#import matplotlib.pyplot as plt

def root_mean_square_log_error(y_true, y_pred):
    epsilon = 1e-15
    return np.sqrt(((np.log(np.clip(y_pred, epsilon, np.inf) + 1.) - np.log(np.clip(y_true, epsilon, np.inf) + 1.)) ** 2).mean())

def cost_trans(y,type):
	if type == 'pow16':
		y = np.power(y,1./16)
	elif type == 'pow11':
		y = np.power(y,1./11)
	elif type == 'pow9':
		y = np.power(y,1./9)
	else:
		y = np.log1p(y)
	return y

def cost_inv_trans(y,type):
	if type == 'pow16':
		y = np.power(y,16)
	elif type == 'pow11':
		y = np.power(y,11)
	elif type == 'pow9':
		y = np.power(y,9)
	else:
		y = np.expm1(y)
	return y

if __name__ == '__main__':

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = False

    X, y, X_submission, ids, cols, cv = data.load(sparse=False)

    clfs = [#Lasso(alpha=0.0075, normalize=True)
            #ExtraTreesRegressor(n_estimators=400, n_jobs=-1),
            xgb.XGBRegressor(max_depth=8, learning_rate=0.019, n_estimators=4000, silent=True, objective='reg:linear', nthread=-1,  max_delta_step=3, min_child_weight=5, subsample=0.83, colsample_bytree=0.62),
            xgb.XGBRegressor(max_depth=7, learning_rate=0.019, n_estimators=4000, silent=True, objective='reg:linear', nthread=-1,  max_delta_step=2, min_child_weight=6, subsample=0.85, colsample_bytree=0.7),
            xgb.XGBRegressor(max_depth=9, learning_rate=0.019, n_estimators=4000, silent=True, objective='reg:linear', nthread=-1,  max_delta_step=2, min_child_weight=5, subsample=0.9, colsample_bytree=0.7)
           ]

    print("Creating train and test sets for blending.")

    trans = ['pow11','pow16', 'pow9']
    scr = np.zeros([len(cv),len(clfs)])
    oob_pred = np.zeros((X.shape[0], len(clfs)))
    #dataset_blend_test_j = np.zeros((X_submission.shape[0], len(cv)))
    for i, (train, test) in enumerate(cv):
	    for j, clf in enumerate(clfs):
	        clf.fit(X[train], cost_trans(y[train], trans[j]))
	        y_submission = cost_inv_trans(clf.predict(X[test]), trans[j])
	        oob_pred[test,j] = y_submission
	        scr[i,j] = root_mean_square_log_error(y[test],y_submission)
	    print("Fold", i, scr[i,:], np.mean(scr[i,:]), root_mean_square_log_error(y[test],oob_pred[test,:].mean(1)))
	    #plt.scatter(y[test], y[test]-y_submission, alpha=0.5)
    score = root_mean_square_log_error(y,oob_pred.mean(1))
    print('total score:', score)
    #print(pd.DataFrame({'col1': cols, 'col2': clf.feature_importances_}).sort('col2'))
    #plt.axis([0, 200, -100, 100])
    #plt.show()

    print("Blending.")
    blend = LinearRegression(fit_intercept=False)
    blend.fit(oob_pred, y)
    print("coefs",blend.coef_)

    print ('Fitting full model')
    sub_pred = np.zeros((X_submission.shape[0], len(clfs)))
    for j, clf in enumerate(clfs):
        clf.fit(X, cost_trans(y, trans[j]))
        y_submission = cost_inv_trans(clf.predict(X_submission), trans[j])
        sub_pred[:,j] = y_submission

    print("Saving Results.")
    preds = pd.DataFrame({"id": ids, "cost": blend.predict(sub_pred)})
    preds.to_csv('blend' + str(score) + '.csv', index=False)

    preds = pd.DataFrame({"id": ids, "cost": sub_pred.mean(1)})
    preds.to_csv('avg' + str(score) + '.csv', index=False)


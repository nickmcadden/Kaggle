import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing
import time
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

def xgb_benchmark_data():
    #data handling, take the input data, and merge them accordingly
    #this is the original data handling routine of the xgb benchark script shared by Gilberto Titericz Junior
    train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
    test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
    tube_data = pd.read_csv('../input/tube.csv')
    bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
    specs_data = pd.read_csv('../input/specs.csv')

    train = pd.merge(train, tube_data, on ='tube_assembly_id')
    train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
    test = pd.merge(test, tube_data, on ='tube_assembly_id')
    test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month

    # drop useless columns and create labels
    idx = test.id.values.astype(int)
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
    labels = train.cost.values

    #for some reason material_id cannot be converted to categorical variable
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

    train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        # print(column_label)
        train[column_label].replace(np.nan,' ', regex=True, inplace= True)
        test[column_label].replace(np.nan,' ', regex=True, inplace= True)

    train.fillna(0, inplace = True)
    test.fillna(0, inplace = True)

    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
            print(i,list(train[1:5,i]) + list(test[1:5,i]))
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])

    return train, test, idx, labels


# pickle data routine in case you saved the data in a local environment
def load_data(pickle_file):
    load_file=open(pickle_file,'rb')
    data=cPickle.load(load_file)
    return  data

# sklearn models
def runmodel(model_type, labels, train, test):
	labels=np.log1p(labels)
	if model_type == "xgboost":
		defined_model = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=50, silent=True, objective='reg:linear', nthread=-1,  min_child_weight=6, subsample=0.86, colsample_bytree=0.5)
	elif model_type == "lasso":
		defined_model = Lasso(alpha = 0.1)
	elif model_type == "ridge":
		defined_model = Ridge(alpha = 0.1)
	elif model_type == "randomforest":
		defined_model = RandomForestRegressor(n_estimators=10, n_jobs=3)
	elif model_type == "extratrees":
		defined_model = ExtraTreesRegressor(n_estimators=50, n_jobs=3)
	elif model_type == "svm":
		defined_model = svm.SVR(C=1.0, epsilon=0.1)
	else:
		printf("You must specify a valid model\n")
	fitted_model = defined_model.fit(train, labels)
	preds=np.expm1(fitted_model.predict(test))
	return  preds

if __name__ == '__main__':
    start_time=time.time()
    test_run=True
    train, test, idx, labels=xgb_benchmark_data()

    # if test run, then perform the cross validation
    if test_run:
        print("perform cross validation")
        model_weights = []
        rnd_state=np.random.RandomState(1)
        for run in range(1, 5):
            train_i, test_i = train_test_split(np.arange(train.shape[0]), train_size = 0.8, random_state = rnd_state )
            tr_train=train[train_i]
            tr_test=train[test_i]
            tr_train_y=labels[train_i]
            tr_test_y=labels[test_i]

            rmse_scores = []
            all_preds = []
            for mod in ['xgboost','randomforest','extratrees']:
                tr_preds = runmodel(mod, tr_train_y, tr_train, tr_test)
                rmse_scores.append((np.sum((np.log1p(tr_preds)-np.log1p(tr_test_y))**2)/len(test_i))**0.5)
                all_preds.append(tr_preds)
            all_preds = np.transpose(np.array(all_preds))
            print(all_preds[0:10], tr_test_y[0:10])
            print ("scores for test run %i" %run)
            print ["%0.6f" % i for i in rmse_scores]
            w = LinearRegression(fit_intercept=False).fit(all_preds,tr_test_y)
            print(w.coef_)
            model_weights.append(w.coef_)
        weights = np.transpose(np.array(model_weights))
        print(weights.mean(axis=1))

    else:
		preds1=runmodel('xgboost', labels, train, test)
		preds2=runmodel('randomforest', labels, train, test)
		preds3=runmodel('extratrees', labels, train, test)
		
		all_preds = pd.DataFrame({"xgb": preds1, "rf": preds2, "ert": preds3})
		preds = (3 * preds1 + preds2 + preds3)/5
		preds = pd.DataFrame({"id": idx, "cost": preds})
		preds.to_csv('xgb-rf-ert.csv', index=False)

    end_time=time.time()
    duration=end_time-start_time
    print ("it takes %.3f seconds"  %(duration))
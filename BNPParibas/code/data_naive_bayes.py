import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import log_loss
import random
from sklearn.naive_bayes import BernoulliNB

def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test, labels):
    features = train.columns
    for col in features:
        if((train[col].dtype == 'object' and col !='v22')):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, labels)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)

    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)

    return train, test
    

def load(m_params):
	train = pd.read_csv("../input/train.csv")
	labels = train['target']
	test = pd.read_csv("../input/test.csv")
	test_ids = test['ID']

	train['v22-1']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
	test['v22-1']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[0]))
	train['v22-2']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))
	test['v22-2']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[1]))
	train['v22-3']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
	test['v22-3']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[2]))
	train['v22-4']=train['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))
	test['v22-4']=test['v22'].fillna('@@@@').apply(lambda x:'@'*(4-len(str(x)))+str(x)).apply(lambda x:ord(x[3]))

	refcols=['v22-1','v22-2','v22-3','v22-4']

	for elt in refcols:
	    if train[elt].dtype=='O':
	        train[elt], temp = pd.factorize(train[elt])
	        test[elt]=temp.get_indexer(test[elt])
	    else:
	        train[elt]=train[elt].round(5)
	        test[elt]=test[elt].round(5)

	train = train.drop(['ID','target','v22'],axis=1).fillna(-999)
	test = test.drop(['ID','v22'],axis=1).fillna(-999)

	print('Munge Data')
	train, test = MungeData(train, test, labels)

	return train.values, labels.values, test.values, test_ids.values

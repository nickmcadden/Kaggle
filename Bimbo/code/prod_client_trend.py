#!/usr/bin/env python
# encoding: utf-8
"""
data2.py

Created by Nick McAdden on 2016-06-18.
Copyright (c) 2016 __MyCompanyName__. All rights reserved.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

print('reading train data...')
train = pd.read_hdf('../input/train.h5', 'train')

train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
	             'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
	             'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
	
#print('reading prod client weeks csv...')
#prod_client_weeks = pd.read_hdf('../input/product_client_weeks.h5','product_client_weeks').values
#prod_client_weeks_cv = pd.read_hdf('../input/product_client_weeks_cv.h5','product_client_weeks_cv').values

print('total prod client week mean_aggs...')
prod_client_mean = train.groupby(['ProductId','ClientId','WeekNum']).agg({'SalesUnitsWeek':np.mean}).reset_index()
print('pivoting... total prod_client_weeks...')
prod_client_weeks = prod_client_mean.pivot_table(index=['ProductId','ClientId'], columns='WeekNum', values='SalesUnitsWeek').reset_index()

prod_client_weeks = prod_client_weeks.values
prod_client_weeks_cv = prod_client_weeks[:,:7]

print(prod_client_weeks[:5])

print('Least squares - cv')
x = np.array([3,4,5,6,7])
X = np.vstack([x, np.ones(len(x))]).T
predscv = np.zeros((prod_client_weeks_cv.shape[0], 2))
print(predscv.shape)
for i in range(prod_client_weeks_cv.shape[0]):
	if i%100000 == 0:
		print i
	y = prod_client_weeks_cv[i,2:7]
	y[np.isnan(y)] = np.nanmean(y)
	m, c = np.linalg.lstsq(X, y)[0]
	predscv[i,:] = np.array([max([m*8+c,1]), max([m*9+c,1])])
	predscv[i,:] = np.array([max([m*8+c,1]), max([m*9+c,1])])

print("Saving Results.")
preds = pd.DataFrame({"ProductId": prod_client_weeks_cv[:,0],
 					  "ClientId": prod_client_weeks_cv[:,1],
					  "week1":predscv[:,0],
					  "week2":predscv[:,1]})

print(preds)
preds.to_csv('../input/cv/prod_client_regression_preds.csv', index=False)

'''
print('Least squares - actual')
x = np.array([3,4,5,6,7,8,9])
X = np.vstack([x, np.ones(len(x))]).T
preds = np.zeros((prod_client_weeks.shape[0], 2))
print(preds.shape)
for i in range(prod_client_weeks.shape[0]):
	if i%100000 == 0:
		print i
	y = prod_client_weeks[i,2:9]
	y[np.isnan(y)] = np.nanmean(y)
	m, c = np.linalg.lstsq(X, y)[0]
	preds[i,:] = np.array([max([m*10+c,1]), max([m*11+c,1])])

print("Saving Results.")
preds = pd.DataFrame({"ProductId": prod_client_weeks[:,0],
 					  "ClientId": prod_client_weeks[:,1],
					  "week1":preds[:,0],
					  "week2":preds[:,1]})

print(preds)
preds.to_csv('../input/prod_client_regression_preds.csv', index=False)
'''
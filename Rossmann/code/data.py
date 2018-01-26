import sys
import pandas as pd
import numpy as np
import os.path
import gc
import tables
import time
import datetime as dt
import pickle
import data
import tabulate
from sklearn.preprocessing import LabelEncoder
from scipy.stats.stats import pearsonr
from scipy.stats import percentileofscore, rankdata

def dateparse(x):
	try:
		return pd.datetime.strptime(x, "%Y-%m-%d")
	except:
		return pd.NaT

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin'] # minimum number of items per bin for a single feature
	getcached = m_params['getcached'] 
	codetest = m_params['codetest'] # this switch will return only 1000 rows of data for testing purposes
	r_seed = m_params['r_seed'] # ensure consistent random seed

	# Read HDF format file
	print("1a. Reading the train and test data...\n")
	
	train = pd.read_csv('input/train.csv', parse_dates=["Date"], date_parser=dateparse)
	test  = pd.read_csv('input/test.csv', parse_dates=["Date"], date_parser=dateparse)
	store  = pd.read_csv('input/store.csv')
	
	train = pd.merge(train,store)
	test = pd.merge(test,store).sort('Id')
	
	print(train.head())

	if codetest:
		train = train.ix[0:999,:]
		test = test.ix[0:999,:]
	
	print("Consider only open stores for training. Closed stores wont count into the score.")
	print("Use only Sales bigger then zero. Simplifies calculation of rmspe")
	train = train[train["Open"] == 1]
	train = train[train["Sales"] != 0]

	labels = train['Sales']
	test_ids = test['Id']
	train_ids = train['Store']
	train.drop(['Sales','Customers','StateHoliday'], axis=1, inplace=True)
	test.drop(['Id','StateHoliday'], axis=1, inplace=True)

	print("1c. Breaking dataframe into numeric, object and date parts...\n")
	train_numeric = train.select_dtypes(include=['float64','int64']).fillna(0)
	test_numeric = test.select_dtypes(include=['float64','int64']).fillna(0)
	
	train_categoric = train.select_dtypes(include=['object']).fillna("_")
	test_categoric = test.select_dtypes(include=['object']).fillna("_")

	train_dates = train.select_dtypes(include=['datetime64[ns]'])
	test_dates = test.select_dtypes(include=['datetime64[ns]'])

	print(train_dates)
	
	''' Deal with categorical data
	print("3. Categorical variable encoding... \n")
	for c in train_categoric.columns:
		freqs = train_categoric[c].append(test_categoric[c]).value_counts()
		train_categoric[c] = pd.match(train_categoric[c].values, freqs[0:1200].index)
		test_categoric[c] = pd.match(test_categoric[c].values, freqs[0:1200].index)
	'''

	print("3. Categorical variable encoding... \n")
	for c in train_categoric.columns:
		print('Encoding',c)
		lbl = LabelEncoder()
		lbl.fit(list(train_categoric.ix[:,c]) + list(test_categoric.ix[:,c]))
		train_categoric.ix[:,c] = lbl.transform(train_categoric.ix[:,c])
		test_categoric.ix[:,c] = lbl.transform(test_categoric.ix[:,c])

	# Date Splits
	train_dates['year']=train_dates['Date'].dt.year
	train_dates['month']=train_dates['Date'].dt.month
	train_dates['day']=train_dates['Date'].dt.day
	test_dates['year']=test_dates['Date'].dt.year
	test_dates['month']=test_dates['Date'].dt.month
	test_dates['day']=test_dates['Date'].dt.day
	
	print(train_dates)

	train_dates.drop(['Date'], axis=1, inplace=True)
	test_dates.drop(['Date'], axis=1, inplace=True)

	print("5. Merging arrays together...\n")

	train = pd.concat([train_categoric, train_dates, train_numeric], axis=1)
	test = pd.concat([test_categoric, test_dates, test_numeric], axis=1)

	print(train.columns, test.columns)

	return train.values, labels.values, test.values, test_ids.values

# New workflow
# Load H5 data
# Get Split data set into object data, numeric data and date data
# Perform data smoothing on all numeric data columns.

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
from scipy.stats.stats import pearsonr

def bin(train_col, test_col, target, minbin=10):
	train_col = train_col.values
	test_col = test_col.values

	long_array = np.concatenate([train_col,test_col],axis=0)
	items, freqs = np.unique(long_array, return_counts = True)

	if len(items) < 10:
		return train_col, test_col

	newitems = np.copy(items)
	binitemcount, ix_start = 0, 0
	max = items[-1]

	# Deal with any special codes depending on correlation with target variable
	if max in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
		f = {'target':['size','mean']}
		corr_table = pd.DataFrame({'item' : train_col, 'target' : target}).groupby('item').agg(f)
		corr_table = corr_table[corr_table['target','size'] > 100]
		corr_table2 = corr_table.ix[max-5:max,:]
		corr_table = corr_table.ix[:max-5,:]
		corr = pearsonr(corr_table.index.values, corr_table['target','mean'])[0]
		for i in corr_table2.index.values.tolist():
			j = corr_table2.ix[i]['target','mean'].astype(float)
			print(i, j, corr)
			if j > 0.34 and corr < 0:
				train_col[train_col == i] = -train_col[train_col == i]
				test_col[test_col == i] = -test_col[test_col == i]
		long_array = np.concatenate([train_col, test_col],axis=0)
		items, freqs = np.unique(long_array, return_counts = True)
		newitems = np.copy(items)

	# Loop through column items and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		binitemcount += freqs[i]
		if binitemcount >= minbin:
			newitems[ix_start:i+1] = ix_start
			binitemcount = 0
			ix_start = i+1
		else:
			newitems[i] = i

	if binitemcount > 0:
		newitems[ix_start:i+1] = len(items) - 1

	# Deal with any special codes depending on correlation with target variable
	#if max in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
	#	print(pd.DataFrame({'freqs': freqs, 'olditem': items, 'newitems': newitems}))

	#print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	train_map_index = np.digitize(train_col, items, right=True)
	test_map_index = np.digitize(test_col, items, right=True)
	return newitems[train_map_index], newitems[test_map_index]

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	codetest = m_params['codetest']

	trainfilename = 'train_' + str(num_features) + str(minbin) + '.h5'
	testfilename = 'test_' + str(num_features) + str(minbin) + '.h5'

	# Read HDF format file
	print("1a. Reading the train and test data...\n")
	if getcached and os.path.isfile('input/'+trainfilename):

		train = pd.read_hdf('input/'+trainfilename, 'train')
		test  = pd.read_hdf('input/'+testfilename, 'test')
		labels = train['target']
		test_ids = test['ID']
		train.drop(['ID','target'], axis=1, inplace=True)
		test.drop(['ID'], axis=1, inplace=True)
	
		return train.values, labels.values, test.values, test_ids.values

	else:		
		train = pd.read_hdf('input/train.h5', 'train')
		test  = pd.read_hdf('input/test.h5','test')
		
		if codetest:
			train = train.ix[0:999,:]
			test = test.ix[0:999,:]

		labels = train['target']
		test_ids = test['ID']
		train_ids = train['ID']
		train.drop(['ID','target'], axis=1, inplace=True)
		test.drop(['ID'], axis=1, inplace=True)

		print("1c. Breaking dataframe into numeric, object and date parts...\n")
		train_numeric = train.select_dtypes(include=['float64','int64'])
		test_numeric = test.select_dtypes(include=['float64','int64'])
		
		train_categoric = train.select_dtypes(include=['object'])
		test_categoric = test.select_dtypes(include=['object'])
		
		train_dates = train.select_dtypes(include=['datetime64[ns]'])
		test_dates = test.select_dtypes(include=['datetime64[ns]'])

		# Zip code engineering
		print("2. Zip code engineering...\n")
		train['VAR_0241'] = train['VAR_0241'].fillna(99999)
		test['VAR_0241'] = test['VAR_0241'].fillna(99999)
		train_zips = np.empty([train.shape[0], 7])
		test_zips = np.empty([test.shape[0], 7])
		try:
			zp = train['VAR_0241'].astype('int64').astype(str)
			zp = zp.replace('','99999')
			train_zips[:,0] = zp.map(lambda x: x[:2]).astype('int32')
			train_zips[:,1] = zp.map(lambda x: x[:1]+x[-1:]).astype('int32')
			train_zips[:,2] = zp.map(lambda x: x[:3]).astype('int32')
			train_zips[:,3] = zp.map(lambda x: x[1:3]).astype('int32')
			train_zips[:,4] = zp.map(lambda x: x[1:4]).astype('int32')
			train_zips[:,5] = zp.map(lambda x: x[2:4]).astype('int32')
			train_zips[:,6] = zp.map(lambda x: x[3:5]).astype('int32')
			zp = test['VAR_0241'].astype('int64').astype(str)
			zp = zp.replace('','99999')
			test_zips[:,0] = zp.map(lambda x: x[:2]).astype('int32')
			test_zips[:,1] = zp.map(lambda x: x[:1]+x[-1:]).astype('int32')
			test_zips[:,2] = zp.map(lambda x: x[:3]).astype('int32')
			test_zips[:,3] = zp.map(lambda x: x[1:3]).astype('int32')
			test_zips[:,4] = zp.map(lambda x: x[1:4]).astype('int32')
			test_zips[:,5] = zp.map(lambda x: x[2:4]).astype('int32')
			test_zips[:,6] = zp.map(lambda x: x[3:5]).astype('int32')
			
			zipcolumns = ['zip0','zip1','zip2','zip3','zip4','zip5','zip6']
			train_zips = pd.DataFrame(train_zips, columns=zipcolumns)
			test_zips = pd.DataFrame(test_zips, columns=zipcolumns)
		except:
			print('Zip codes cant be encoded')
			exit()

		# Deal with categorical data
		print("3. Categorical variable encoding... \n")
		for c in train_categoric.columns:
			freqs = train_categoric[c].append(test_categoric[c]).value_counts()
			train_categoric[c] = pd.match(train_categoric[c].values, freqs[0:70].index)
			test_categoric[c] = pd.match(test_categoric[c].values, freqs[0:70].index)

		# Deal with categorical data
		print("4. Numeric Column Smoothing... \n")
		train_numeric = train_numeric.fillna(0)
		test_numeric = test_numeric.fillna(0)
		numeric_col_count = 0
		if minbin > 1:
			for c in train_numeric.columns:
				train_numeric[c], test_numeric[c] = bin(train_numeric[c], test_numeric[c], labels, minbin)
				numeric_col_count += 1
				if not (numeric_col_count % 10):
					print('Numeric Col Count: ', numeric_col_count)

		gc.collect()

		# Create new date transformations
		print('5. Create new date columns...\n')
		def tdtoint(td):
			if not pd.isnull(td):
				return td.astype('timedelta64[D]').astype(np.int32)
			else:
				return 0

		# Diffs between important dates
		for i in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
			for j in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
				if i < j:
					keypair = i+'_'+j
				else:
					keypair = j+'_'+i
				if i!=j and keypair not in train_dates.columns:
					train_dates[keypair] = train_dates[i] - train[j]
					train_dates[keypair] = train_dates[keypair].apply(tdtoint)
					test_dates[keypair] = test_dates[i] - test_dates[j]
					test_dates[keypair] = test_dates[keypair].apply(tdtoint)

		# Date Splits
		datecols = pd.read_pickle('input/datecols.pkl')
		for c in datecols['col'].values.tolist():
			train_dates[c+'_y']=train_dates[c].dt.year
			train_dates[c+'_m']=train_dates[c].dt.month
			train_dates[c+'_d']=train_dates[c].dt.day
			train_dates[c+'_wd']=train_dates[c].dt.weekday
			train_dates[c+'_hr']=train_dates[c].dt.hour
			test_dates[c+'_y']=test_dates[c].dt.year
			test_dates[c+'_m']=test_dates[c].dt.month
			test_dates[c+'_d']=test_dates[c].dt.day
			test_dates[c+'_wd']=test_dates[c].dt.weekday
			test_dates[c+'_hr']=test_dates[c].dt.hour

		train_dates.drop(datecols['col'].values.tolist(), axis=1, inplace=True)
		test_dates.drop(datecols['col'].values.tolist(), axis=1, inplace=True)

		gc.collect()

	print("5. Merging arrays together...\n")
	# put seperate parts together again

	train = pd.concat([train_categoric, train_dates, train_numeric, train_zips], axis=1)
	test = pd.concat([test_categoric, test_dates, test_numeric, test_zips], axis=1)

	# Get only top n features
	print("1b. Filtering by pickled important columns...\n")
	cols = pd.read_pickle("input/vars_importance.pkl")
	cols = list(cols.ix[0:num_features,"var"])
	
	for c in cols:
		if c not in train.columns:
			cols.remove(c)

	train = train[cols].fillna(0)
	test = test[cols].fillna(0)

	gc.collect()
	
	try:
		print("6. Writing to hdf format...\n")
		pd.concat([train_ids, train, labels], axis=1).to_hdf('input/' + trainfilename, key='train', format='fixed', mode='w')
		pd.concat([test_ids, test], axis=1).to_hdf('input/' + testfilename, key='test', format='fixed', mode='w')
	except:
		error = sys.exc_info()[0]
		print( "Error: %s" % error )

	return train.values, labels.values, test.values, test_ids.values

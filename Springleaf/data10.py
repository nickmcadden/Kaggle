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
from scipy.stats import percentileofscore, rankdata

def bin(train_col, test_col, target, minbin=30, minitems=10, dynamic = True, dynamic_mult = 2, bytarget=True,  addnoise=False):
	# for each feature
	# 1. Append test and train data
	# 2. Get ordered table of unique items with frequencies and target variable averages for all unique items
	# 4. Loop through item table and for each item if frequency is less than minbin, group with next item assigning new value to new groups
	# 5. Store new target value for new group if required.
	# 6. If ordering by target sort dataframe on new target rate and re-assign bucket values on target rank

	long_array = pd.concat([train_col,test_col], axis=0)
	items =  pd.DataFrame(pd.value_counts(long_array, ascending = True).reset_index())
	items.columns = ['item','freq']

	if len(items) < minitems:
		return train_col, test_col

	newitems = np.empty(len(items), dtype=np.float64)
	newtargets = np.empty(len(items), dtype=np.float64)
	binitemcount, trainitemtargets, trainitemcount, ix_start = 0, 0, 0, 0

	new_long_array = np.copy(long_array)

	# dynamic bin sizing will ensure minimum size of the bin is proportional to the number of unique items for that feature
	if dynamic == True:
		minbin = int(np.sqrt(len(items)) * dynamic_mult)

	# We can add noise to the values of each item for increased variance/smoothing effect 
	if addnoise == True:
		for i in range(0, len(items)):
			ixs = new_long_array[new_long_array==items[i]]
			guassian_noise = np.random.normal(0, 1, len(ixs))
			new_long_array[ixs] == new_long_array[ixs] + guassian_noise

	# If using the target correlation to provide a natural ordering to the data this is done here
	if bytarget == True:
		f = {'target':['sum','mean','count']}
		corr = pd.DataFrame({'item' : train_col, 'target' : target}).groupby('item', sort=False, as_index=False).agg(f)
		corr = corr.fillna(0)
		items = pd.merge(corr, items, how='right', on='item').sort('item')
		items.columns = ['item','item2','traintarget','trainmean','trainfreq','freq']
		items.drop('item2', axis=1, inplace=True)
		items.index = range(0, len(items))

	# Loop through column items and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		binitemcount += items.ix[i,'freq']
		trainitemcount += items.ix[i,'trainfreq']
		trainitemtargets += items.ix[i,'traintarget']
			
		if binitemcount >= minbin:
			newitems[ix_start:i+1] = ix_start
			newtargets[ix_start:i+1] = trainitemtargets / trainitemcount
			if items.ix[i,'freq'] > minbin:
				newitems[i] = i
				newtargets[i] = items.ix[i,'traintarget'] / items.ix[i,'trainfreq']
			binitemcount, trainitemtargets, trainitemcount = 0, 0, 0
			ix_start = i+1
		else:
			newitems[i] = ix_start

	if binitemcount > 0:
		newitems[ix_start:i+1] = len(items) - 1
		newtargets[ix_start:i+1] = trainitemtargets / trainitemcount

	newtargets = np.nan_to_num(newtargets)

	if bytarget == True:
		newitems = rankdata(newtargets, method='min')

	allnewitems = pd.DataFrame({'item':items['item'].values, 'newitem': newitems, 'freq':items['freq'].values, 'target':newtargets})
	allnewitems = allnewitems.sort('newitem')

	'''
	with pd.option_context('display.max_rows', 1000):
		print(allnewitems)
	exit()
	'''

	#print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	train_map_index = np.digitize(train_col, items['item'], right=True)
	test_map_index = np.digitize(test_col, items['item'], right=True)
	return newitems[train_map_index], newitems[test_map_index]

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin'] # minimum number of items per bin for a single feature
	getcached = m_params['getcached'] 
	codetest = m_params['codetest'] # this switch will return only 1000 rows of data for testing purposes
	r_seed = m_params['r_seed'] # ensure consistent random seed

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
			train = train.ix[0:9999,:]
			test = test.ix[0:9999,:]

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

	gc.collect()

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
		print("Error: %s" % error)

	return train.values, labels.values, test.values, test_ids.values

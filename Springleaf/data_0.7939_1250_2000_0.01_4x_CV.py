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

def bin(col, traincol, target, minbin=10):
	colname = col.name
	col = col.values
	col = np.nan_to_num(col)
	target = target.values

	items, freqs = np.unique(col, return_counts = True)
	
	if len(items)==1:
		return col

	newitems = np.copy(items)
	binitemcount, cumvaluesum = 0, 0

	max = items[-1]
	ix_start, ix_end = 0, 0

	# Deal with any special codes depending on correlation with target variable
	if max in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
		traincol = traincol.values
		targets0 = np.sum((traincol==items[1]) * target)
		corr0 = float(targets0) / float(freqs[1])
		#print(colname, freqs[0], targets0, corr0)
		if corr0 > 0.34:
			col[col >= max-4] = -col[col >= max-4]
			items, freqs = np.unique(col, return_counts = True)
			newitems = np.copy(items)

	# Loop through column values and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		binitemcount += freqs[i]
		if binitemcount >= minbin:
			newitems[ix_start:i+1] = ix_start
			binitemcount = 0
			ix_start = i+1
		else:
			newitems[i] = i

	if binitemcount > 0:
		newitems[ix_start:i+1] = len(items)-1

	# Deal with any special codes depending on correlation with target variable
	if max in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
		print(pd.DataFrame({'freqs': freqs, 'olditem': items, 'newitems': newitems}))

	'''
	# Loop through column values and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		if freqs[i] >= minbin:
			newitems[i] = i
			ix_start = i+1 
		else:
			if binitemcount >= minbin:
				newitems[ix_start:ix_end+1] = ix_start
				binitemcount = 0
				ix_start = i 
			ix_end = i
			binitemcount += freqs[i]

	if binitemcount > 0:
		newitems[ix_start:ix_end+1] = len(items)-1


	# Loop through column values and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		if freqs[i] >= minbin:
			ix_start = i+1 
		else:
			if binitemcount >= minbin:
				newitems[ix_start:ix_end+1] = round(cumvaluesum / binitemcount)
				binitemcount, cumvaluesum = 0, 0
				ix_start = i 
			ix_end = i
			binitemcount += freqs[i]
			cumvaluesum += freqs[i] * items[i]
	
	# Mop up any rows which haven't been binned
	if binitemcount > 0:
		newitems[ix_start:ix_end+1] = round(np.sum(freqs[ix_start: ix_end+1] * items[ix_start: ix_end+1])/ binitemcount)
	'''

	# print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	map_index = np.digitize(col, items, right=True)
	return newitems[map_index]

def load(m_params):

	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	
	t0 = time.time()

	trainfilename = 'train_' + str(num_features) + str(minbin) + '.h5'
	testfilename = 'test_' + str(num_features) + str(minbin) + '.h5'

	# Read HDF format file
	print("1. Reading the train and test data...\n")
	
	if getcached and os.path.isfile(trainfilename):
		
		train = pd.read_hdf(trainfilename, 'train')
		test  = pd.read_hdf(testfilename, 'test')
		labels = train['target']
		test_ids = test['ID']
		train.drop(['ID','target'], axis=1, inplace=True)
		test.drop(['ID'], axis=1, inplace=True)
		
		return train.values, labels.values, test.values, test_ids.values
		
	elif getcached and os.path.isfile('train_binned_' + str(minbin) + '.h5'):

		train = pd.read_hdf('train_binned_' + str(minbin) + '.h5', 'train')
		test  = pd.read_hdf('test_binned_' + str(minbin) + '.h5', 'test')
		labels = train['target']
		test_ids = test['ID']
		train.drop(['ID','target'], axis=1, inplace=True)
		test.drop(['ID'], axis=1, inplace=True)

	else:

		train = pd.read_hdf('train.h5', 'train')
		test  = pd.read_hdf('test.h5','test')
	
		labels = train['target']
		test_ids = test['ID']

		gc.collect()
		
		print("Postcode column \n")
		print(train['VAR_0241'].dtype, len(np.unique(train['VAR_0241'])))
		print(test['VAR_0241'].dtype, len(np.unique(test['VAR_0241'])))
		
		# Zip code engineering
		print("4. Zip code engineering...\n")
		train['VAR_0241'] = train['VAR_0241'].fillna(99999)
		test['VAR_0241'] = test['VAR_0241'].fillna(99999)
		try:
			zp = train['VAR_0241'].astype('int64').astype(str)
			zp = zp.replace('','99999')
			train['zip_00xxx'] = zp.map(lambda x: x[:2]).astype('int32')
			train['zip_0xxx0'] = zp.map(lambda x: x[:1]+x[-1:]).astype('int32')
			train['zip_000xx'] = zp.map(lambda x: x[:3]).astype('int32')
			train['zip_x00xx'] = zp.map(lambda x: x[1:3]).astype('int32')
			train['zip_x000x'] = zp.map(lambda x: x[1:4]).astype('int32')
			train['zip_xx00x'] = zp.map(lambda x: x[2:4]).astype('int32')
			train['zip_xxx00'] = zp.map(lambda x: x[3:5]).astype('int32')
			zp = test['VAR_0241'].astype('int64').astype(str)
			zp = zp.replace('','99999')
			test['zip_00xxx'] = zp.map(lambda x: x[:2]).astype('int32')
			test['zip_0xxx0'] = zp.map(lambda x: x[:1]+x[-1:]).astype('int32')
			test['zip_000xx'] = zp.map(lambda x: x[:3]).astype('int32')
			test['zip_x00xx'] = zp.map(lambda x: x[1:3]).astype('int32')
			test['zip_x000x'] = zp.map(lambda x: x[1:4]).astype('int32')
			test['zip_xx00x'] = zp.map(lambda x: x[2:4]).astype('int32')
			test['zip_xxx00'] = zp.map(lambda x: x[3:5]).astype('int32')
		except:
			print('BOLLOCKS Zip codes cant be encoded')
			exit()

		# Deal with categorical data and smoothing
		print("2. Categorical variable encoding and numeric col smoothing... \n")
		numeric_col_count = 0
		for c in train.columns[1:-1]:
			if train[c].name != 'target':
				if train[c].dtype.name == 'object':
					freqs = train[c].append(test[c]).value_counts()
					train[c] = pd.match(train[c].values, freqs[0:70].index)
					test[c] = pd.match(test[c].values, freqs[0:70].index)
				elif train[c].dtype.name in ['int64', 'float64'] and minbin > 1:
					# smooth numeric cols
					train[c] = bin(train[c], train[c], train['target'], minbin)
					test[c] = bin(test[c], train[c], train['target'], minbin)
					numeric_col_count += 1
					if not (numeric_col_count % 10):
						print('Numeric Col Count: ', numeric_col_count)

		gc.collect()

		# Create new date transformations
		print('3. Create new date columns...\n')
		def tdtoint(td):
			if not pd.isnull(td):
				return td.astype('timedelta64[D]').astype(int)
			else:
				return 0

		# Diffs between important dates
		for i in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
			for j in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
				if i < j:
					keypair = i+'_'+j
				else:
					keypair = j+'_'+i
				if i!=j and keypair not in train.columns:
					train[keypair] = train[i] - train[j]
					train[keypair] = train[keypair].apply(tdtoint)
					test[keypair] = test[i] - test[j]
					test[keypair] = test[keypair].apply(tdtoint)

		# Date Splits
		datecols = pd.read_pickle('datecols.pkl')
		for c in datecols['col'].values.tolist():
			train[c+'_y']=train[c].dt.year
			train[c+'_m']=train[c].dt.month
			train[c+'_d']=train[c].dt.day
			train[c+'_wd']=train[c].dt.weekday
			train[c+'_hr']=train[c].dt.hour
			test[c+'_y']=test[c].dt.year
			test[c+'_m']=test[c].dt.month
			test[c+'_d']=test[c].dt.day
			test[c+'_wd']=test[c].dt.weekday
			test[c+'_hr']=test[c].dt.hour

		train.drop(datecols['col'].values.tolist(), axis=1, inplace=True)

		gc.collect()

		# Fill any remaining N/As
		train = train.fillna(0)
		test = test.fillna(0)
	
		#print("4.5. Writing to hdf format...\n")
		#train.to_hdf('train_binned_' + str(minbin) + '.h5',key='train',format='fixed',mode='w')
		#test.to_hdf('test_binned_' + str(minbin) + '.h5',key='test',format='fixed',mode='w')

	# Get only top n features
	print("5. Filtering by pickled important columns...\n")
	cols = pd.read_pickle("vars_importance.pkl")
	cols = cols.ix[0:num_features,"var"].tolist()
	
	print("6. Writing to hdf format...\n")
	#zipcols = ['zip_00xxx', 'zip_0xxx0', 'zip_000xx', 'zip_x00xx', 'zip_x000x', 'zip_xx00x', 'zip_xxx00']
	zipcols = []
	train[cols+zipcols+['ID','target']].to_hdf(trainfilename,key='train',format='fixed',mode='w')
	test[cols+zipcols+['ID']].to_hdf(testfilename,key='test',format='fixed',mode='w')

	train = train[cols+zipcols]
	test = test[cols+zipcols]

	gc.collect()

	return train.values, labels.values, test.values, test_ids.values

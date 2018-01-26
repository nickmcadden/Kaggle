import numpy as np
cimport cython
cimport numpy as np

def bin(np.ndarray[long] train_col, 
		np.ndarray[long] test_col, 
		np.ndarray[long] target, 
		unsigned int minbin=10):

	cdef np.ndarray[long] items, newitems
	cdef np.ndarray[long] freqs, train_map_index, test_map_index
	cdef np.ndarray[long] long_array = np.concatenate([train_col,test_col], axis=0)
	cdef unsigned int binitemcount = 0
	cdef unsigned int ixstart = 0
	cdef long maxval
	cdef int train_targets
	cdef unsigned int i
	cdef unsigned int unique_vals
	cdef double corr

	items, freqs = np.unique(long_array, return_counts = True)
	
	unique_vals = len(items)
	
	if unique_vals == 1:
		return train_col, test_col

	newitems = np.copy(items)
	binitemcount, ix_start = 0, 0
	maxval = items[-1]

	# Deal with any special codes depending on correlation with target variable
	if maxval in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
		train_targets = np.sum((train_col == items[1]) * target)
		corr = float(train_targets) / (float(freqs[1])/2)
		if corr > 0.34:
			train_col[train_col >= maxval-4] = -train_col[train_col >= maxval-4]
			test_col[test_col >= maxval-4] = -test_col[test_col >= maxval-4]
			long_array = np.concatenate([train_col,test_col],axis=0)
			items, freqs = np.unique(long_array, return_counts = True)
			newitems = np.copy(items)

	# Loop through column items and group any consecutive values occuring less frequently than minimum bin size
	for i in range(unique_vals):
		binitemcount += freqs[i]
		if binitemcount >= minbin:
			newitems[ix_start:i+1] = ix_start
			binitemcount = 0
			ix_start = i+1
		else:
			newitems[i] = i

	if binitemcount > 0:
		newitems[ix_start:i+1] = unique_vals - 1

	# Deal with any special codes depending on correlation with target variable
	#if max in [999,9999,99999,999999,9999999,99999999,999999999,9999999999]:
	#	print(pd.DataFrame({'freqs': freqs, 'olditem': items, 'newitems': newitems}))

	#print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	train_map_index = np.digitize(train_col, items, right=True)
	test_map_index = np.digitize(test_col, items, right=True)
	return newitems[train_map_index], newitems[test_map_index]
#!/usr/bin/env python
# encoding: utf-8
"""
bin.py

Created by Nick McAdden on 2016-03-26.
Copyright (c) 2016 __MyCompanyName__. All rights reserved.
"""

import pandas as pd
import numpy as np

def bin(train_col, test_col, target, minbin=10):

	long_array = np.concatenate([train_col,test_col], axis=0)
	items, freqs = np.unique(long_array, return_counts = True)

	print(items)
	print(freqs)

	if len(items)==1:
		return train_col, test_col

	newitems = np.copy(items)
	binitemcount, ix_start = 0, 0

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

	#print(pd.DataFrame({'freqs': freqs, 'olditem': items, 'newitems': newitems}))
	#print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	train_map_index = np.digitize(train_col, items, right=True)
	test_map_index = np.digitize(test_col, items, right=True)
	return newitems[train_map_index], newitems[test_map_index]

traincol = np.array([1.2,3.3,6.2,3,4,3,2,4,3,4,5,4,3,2,3,4,3,4,5,4,3,2,4,5])
testcol = np.array([1,3,2,3,4,3,2.2,4.3,3,4,5,4,1,1,1,2,3,5,3,5,3,5,3,3,])
labels = np.array([1,1,1,1,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,0,0,0,1,0,])

trnnew, tstnew = bin(traincol, testcol, labels, 20)
print(traincol)
print(trnnew)
print(testcol)
print(tstnew)


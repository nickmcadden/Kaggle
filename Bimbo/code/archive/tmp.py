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

print('reading train data...')
pcrp = pd.read_csv('../input/prod_client_regression_preds.csv')

pcrp_cv = pcrp.copy()

pcrp.drop(['week8','week9'], axis=1, inplace=True)
pcrp_cv.drop(['week10','week11'], axis=1, inplace=True)

print("Saving Results.")
pcrp.to_csv('../input/prod_client_regression_preds.csv', index=False)
pcrp_cv.to_csv('../input/cv/prod_client_regression_preds.csv', index=False)


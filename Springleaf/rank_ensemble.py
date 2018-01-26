#!/usr/bin/env python
# encoding: utf-8
"""
Created by Nick McAdden on 2015-10-11.
Copyright (c) 2015 __MyCompanyName__. All rights reserved.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata

weights = [0.2,0.6,0.9,1.0] # your weights for each model
files = ['xgb_8000_1888_0.002.csv','xgb___080187.csv', 'xgb___080200.csv', 'xgb___080203.csv'] # your prediction files 

finalRank = 0
for i in range(len(files)):
    temp_df = pd.read_csv('output/'+files[i])
    finalRank = finalRank + rankdata(temp_df.target, method='ordinal') * weights[i]
finalRank = finalRank / (max(finalRank) + 1.0)

df = temp_df.copy()
df['target'] = finalRank
df.to_csv('output/rank_ensemble1.csv', index = False)


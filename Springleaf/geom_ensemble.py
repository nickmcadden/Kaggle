#!/usr/bin/env python
# encoding: utf-8
"""
Created by Nick McAdden on 2015-10-11.
Copyright (c) 2015 All rights reserved.
"""

import pandas as pd
import numpy as np

weights = [0.1,0.7,0.9,1.0] # your weights for each model
files = ['xgb_8000_1888_0.002.csv','xgb___080187.csv', 'xgb___080200.csv', 'xgb___080203.csv'] # your prediction files 

finalScore = 0
for i in range(len(files)):
    temp_df = pd.read_csv('output/'+files[i])
    finalScore = finalScore + np.log(temp_df.target) * weights[i]
finalScore = np.exp(finalScore / sum(weights))

df = temp_df.copy()
df['target'] = finalScore
df.to_csv('output/geom_ensemble.csv', index = False)


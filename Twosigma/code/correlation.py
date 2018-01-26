import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

print("reading the data\n")
pred1 = pd.read_csv('../output/nnet_blend_all.csv').sort_values(by=["listing_id"])[["high", "medium", "low"]]
pred2 = pd.read_csv('../output/pred_xgb_blend_all_420.csv').sort_values(by=["listing_id"])[["high", "medium", "low"]]

print pred1[1:10]
print pred2[1:10]

for i in range(3):
	print(i, pearsonr(pred1.iloc[:, i], pred2.iloc[:, i]))

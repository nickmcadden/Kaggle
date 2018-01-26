import sys
import pandas as pd
import numpy as np

print("reading the data\n")
test  = pd.read_csv('../input/test.csv')
pred1 = pd.read_csv('../output/pred_nnet_blend.csv')
pred2 = pd.read_csv('../output/pred_xgb_blend_1802.csv')

print pred1[1:10]
print pred2[1:10]

pred = pred1["PredictedProb"]*0.5 + pred2["PredictedProb"]*0.5

print("Blending.")

preds = pd.DataFrame({"ID": test['ID'].values, "PredictedProb": pred.values})
preds.to_csv('../output/xgb_nnet_blend_avg_v2' + '.csv', index=False)
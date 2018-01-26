import sys
import pandas as pd
import numpy as np

print("reading the data\n")
test  = pd.read_csv('../input/test.csv')
pred = pd.read_csv('../output/xgb_nnet_blend_avg.csv')

print pred[50:100]

pred['PredictedProb'] = np.power(pred['PredictedProb'],1/(pred['PredictedProb']+0.5))

print("Blending.")
print pred[50:100]

preds = pd.DataFrame({"ID": test['ID'].values, "PredictedProb": pred['PredictedProb'].values})
preds.to_csv('../output/test_blend_avg' + '.csv', index=False)
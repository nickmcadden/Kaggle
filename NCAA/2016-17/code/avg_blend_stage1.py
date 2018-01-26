import sys
import pandas as pd
import numpy as np

print("reading the data\n")
sample_sub  = pd.read_csv('../input/sample_submission_stage1.csv')
pred1 = pd.read_csv('../output/dixoncoles_stage1.csv')
pred2 = pd.read_csv('../output/lr_stage1.csv')
pred3 = pd.read_csv('../output/lr2_stage1.csv')
pred4 = pd.read_csv('../output/lr3_stage1.csv')

print pred1[1:10]
print pred2[1:10]
print pred3[1:10]

pred = pred1["pred"] * 0.5 + pred2["pred"] * 0.2 + pred3["pred"] * 0.1 + pred4["pred"] * 0.2
print(pred[1:10])

print("Blending.")

preds = pd.DataFrame({"id": sample_sub['id'].values, "pred": pred.values})
preds.to_csv('../output/lr123_dc_blend' + '.csv', index=False)

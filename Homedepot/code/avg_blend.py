import sys
import pandas as pd
import numpy as np

print("reading the data\n")
pred1 = pd.read_csv('../output/homedepot_freq.csv')
pred2 = pd.read_csv('../output/sub_pred_rfentropy_0.465155875942.csv')
pred3  = pd.read_csv('../output/homedepot_scr.csv')
pred4 = pd.read_csv('../output/homedepot_scrfreqpoisson_ng5_uid.csv')
pred5 = pd.read_csv('../output/larko_0473_submission.csv')
pred6 = pd.read_csv('../output/sub_pred_xgb_0.470647963443.csv')
pred7 = pd.read_csv('../output/owlsubmission.csv')

print pred1[1:10]
print pred2[1:10]

pred = pred1["relevance"]*0.03 + pred2["relevance"]*0.12 + pred3["relevance"]*0.03 + pred4["relevance"]*0.10 + pred5["relevance"]*0.12 + pred6["relevance"]*0.20 + pred7["relevance"]*0.4
print("Blending.")

preds = pd.DataFrame({"id": pred1['id'].values, "relevance": pred.values})
preds.to_csv('../output/5model_avg6' + '.csv', index=False)
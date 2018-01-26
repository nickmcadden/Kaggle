import sys
import pandas as pd
import numpy as np

print("reading the data\n")
pred1 = pd.read_csv('../output/sub_pred_xgb_poisson0.470876882477.csv')
pred2 = pd.read_csv('../output/sub_pred_xgb_0.472169801807.csv')
pred3  = pd.read_csv('../output/sub_pred_rfentropy_0.470885565946.csv')
pred4 = pd.read_csv('../output/sub_pred_etrentropy_0.47619222738.csv')
pred5 = pd.read_csv('../output/homedepot_scrfreqpoisson_ng5_uid.csv')


print pred1[1:10]
print pred2[1:10]

pred = pred1["relevance"]*0.3 + pred2["relevance"]*0.15 + pred3["relevance"]*0.3 + pred4["relevance"]*0.05 + pred5["relevance"]*0.2
print("Blending.")

preds = pd.DataFrame({"id": pred1['id'].values, "relevance": pred.values})
preds.to_csv('../output/5model_avg5' + '.csv', index=False)
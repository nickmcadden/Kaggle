import sys
import pandas as pd
import numpy as np

print("reading the data\n")
pred1 = pd.read_csv('../output/xgb_cvstats1.csv').sort_values(by=["listing_id"]).reset_index()
pred2 = pd.read_csv('../output/xgb_cvstats2.csv').sort_values(by=["listing_id"]).reset_index()
pred3 = pd.read_csv('../output/xgb_cvstats3.csv').sort_values(by=["listing_id"]).reset_index()
pred4 = pd.read_csv('../output/xgb_cvstats4.csv').sort_values(by=["listing_id"]).reset_index()
pred5 = pd.read_csv('../output/xgb_cvstats5.csv').sort_values(by=["listing_id"]).reset_index()
pred6 = pd.read_csv('../output/xgb_cvstats6.csv').sort_values(by=["listing_id"]).reset_index()

print pred1[1:10]
print pred2[1:10]

pred = pd.DataFrame({"listing_id": pred1['listing_id'].values})
pred[["high","medium","low"]] = (pred1[["high","medium","low"]] + pred2[["high","medium","low"]] + pred3[["high","medium","low"]] + pred4[["high","medium","low"]] + pred5[["high","medium","low"]] + pred6[["high","medium","low"]]) / 6
print("Blending.")
print pred[1:10]

#preds = pd.DataFrame({"id": pred1['id'].values, "relevance": pred.values})
pred.to_csv('../output/xgb_cvstats_blend' + '.csv', index=False)

import sys
import pandas as pd
import numpy as np

print("reading the data\n")
pred1 = pd.read_csv('../output/xgb_blend20.csv').sort_values(by=["listing_id"]).reset_index()
pred2 = pd.read_csv('../output/xgb_blend21.csv').sort_values(by=["listing_id"]).reset_index()

print pred1[1:10]
print pred2[1:10]

pred = pd.DataFrame({"listing_id": pred1['listing_id'].values})
pred[["high","medium","low"]] = (pred1[["high","medium","low"]]*0.5 + pred2[["high","medium","low"]]*0.5)
print("Blending.")
print pred[1:10]

#preds = pd.DataFrame({"id": pred1['id'].values, "relevance": pred.values})
pred.to_csv('../output/xgb_blend22' + '.csv', index=False)

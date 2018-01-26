import sys
import pandas as pd
import numpy as np

print("reading the data\n")
pred1 = pd.read_csv('../output/xgb_blend20.csv').sort_values(by=["listing_id"]).reset_index()
pred = pd.DataFrame({"listing_id": pred1['listing_id'].values})

print(pred1[1:10])

over50pc = (pred1[["high", "medium", "low"]] >= 0.5).astype(int)
under50pc = (pred1[["high", "medium", "low"]] < 0.5).astype(int)

temp1 = np.power((pred1[["high", "medium", "low"]] - 0.5) * 2, 0.8)/2 +0.5
temp2 = 1- (np.power(((1-pred1[["high", "medium", "low"]]) - 0.5) * 2, 0.8)/2 +0.5)
temp1[np.isnan(temp1)] = 0
temp2[np.isnan(temp2)] = 0

pred[["high","medium","low"]] = temp2 * under50pc + temp1 * over50pc
print("Blending.")
print(pred[1:10])

#preds = pd.DataFrame({"id": pred1['id'].values, "relevance": pred.values})
pred.to_csv('../output/xgb_blend23' + '.csv', index=False)

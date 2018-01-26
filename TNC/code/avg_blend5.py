import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model1 = pd.read_csv('../output/tnc_vgs_no_alb.csv')
model2 = pd.read_csv('../output/tnc_vgs_only_alb.csv')
model_avg = model1

model_avg.iloc[:,1:] = model_avg.iloc[:,1:].multiply(model2.iloc[:,2], axis=0)
model_avg.iloc[:,1] = model2.iloc[:,1]

row_sums = np.sum(model_avg.iloc[:,1:], axis=1)
model_avg.iloc[:,1:] = np.array(model_avg.iloc[:,1:]) / row_sums[: , None]

print("Saving Results.")
model_pathname = '../output/tnc_vgs_no_alb_alb'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')

import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = ['tnc_vgs_no_alb_alb.csv',
				'tnc_vgs_sub_3.csv',
				'tnc_vgs_corrected.csv',
				'tnc_vgs_jitter.csv']

weight = [1.0/4,1.0/4,1.0/4,1.0/4]

for i, model_name in enumerate(model_output):
	model = pd.read_csv('../output/' + model_name)
	if i==0:
		model_avg = model
		model_avg.iloc[:,1:] = model.iloc[:,1:] * weight[i]
	else:
		model_avg.iloc[:,1:] += model.iloc[:,1:] * weight[i]

row_sums = np.sum(model_avg.iloc[:,1:], axis=1)
model_avg.iloc[:,1:] = np.array(model_avg.iloc[:,1:]) / row_sums[: , None]

print("Saving Results.")
model_pathname = '../output/tnc_vgs_blend2'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')

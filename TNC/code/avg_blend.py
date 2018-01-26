import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = [
				'submission_vgs_0.csv',
				'submission_vgs_1.csv',
				'submission_vgs_2.csv',
				'submission_vgs_3.csv',
				'submission_vgs_4.csv',
				'submission_vgs_5.csv',
				'submission_vgs_6.csv',
				'submission_vgs_7.csv']

weight = [1.0/len(model_output)]*len(model_output)

for i, model_name in enumerate(model_output):
	model = pd.read_csv('../output/' + model_name)
	if i==0:
		model_avg = model
		model_avg.iloc[:,1:] = model.iloc[:,1:] * weight[i]
	else:
		model_avg.iloc[:,1:] += model.iloc[:,1:] * weight[i]

print("Saving Results.")
model_pathname = '../output/tnc_vgs_only_alb'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')

import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = ['tnc_resnet50_pseudo_zoomed.csv',
				'tnc_vgg16_pseudo_zoomed.csv',
				'tnc_vgg19_pseudo_zoomed.csv',
				'tnc_vgs_pseudo_zoomed.csv']

weight = [0.30,0.20,0.20,0.30]

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
model_pathname = '../output/tnc_pseudo_zoomed_blend'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')

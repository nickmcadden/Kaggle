import pandas as pd
import numpy as np

print("reading test data\n")
#train = pd.read_csv('train.csv', parse_dates=datecols, date_parser=dateparse)
test  = pd.read_csv('test.csv')

ids = test["ID"]
ids.to_csv('test_ids.csv', index=False)

sub = pd.read_csv('xgb.csv')
sub['ID']=ids

sub.to_csv('xgb.csv', index=False)

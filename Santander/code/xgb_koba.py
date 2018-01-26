import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import xgboost as xgb
import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# clean and split data

# remove constant columns (std = 0)
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1,len(cols)):
        if np.array_equal(v,train[cols[j]].values):
            remove.append(cols[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# split data into train and test
test_id = test.ID
test = test.drop(["ID"],axis=1)

X = train.drop(["TARGET","ID"],axis=1)
y = train.TARGET.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)
print(X_train.shape, X_test.shape, test.shape)

## # Feature selection
clf = ExtraTreesClassifier(random_state=1729)
selector = clf.fit(X_train, y_train)

# plot most important features
feat_imp = pd.Series(clf.feature_importances_, index = X_train.columns.values).sort_values(ascending=False)

# clf.feature_importances_ 
#fs = SelectFromModel(selector, prefit=True)

#X_train = fs.transform(X_train)
#X_test = fs.transform(X_test)
#test = fs.transform(test)

print(X_train.shape, X_test.shape, test.shape)

dtrain = xgb.DMatrix(X_train, y_train)
dval =	xgb.DMatrix(X_test, y_test)

xgb_param = {'silent' : 1, 'max_depth' : 5, 'eval_metric': 'auc', 'objective': 'reg:logistic', 'eta':0.01, 'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.5}
m2_xgb = xgb.train(xgb_param, dtrain, 110, evals=([dtrain,'train'], [dval,'val']))
## # Train Model
# classifier from xgboost
#m2_xgb = xgb.XGBClassifier(n_estimators=110, nthread=-1, max_depth = 4, seed=1)
#m2_xgb.fit(X_train, y_train, eval_metric="auc", verbose = False, eval_set=[(X_test, y_test)])

dtest = xgb.DMatrix(test)

# calculate the auc score
#print("Roc AUC: ", roc_auc_score(y_test, m2_xgb.predict(X_test), average='macro'))            
## # Submission
probs = m2_xgb.predict(dtest)

submission = pd.DataFrame({"ID":test_id, "TARGET": probs})
submission.to_csv("../output/subkoba.csv", index=False)
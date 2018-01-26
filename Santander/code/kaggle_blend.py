from __future__ import division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

if __name__ == '__main__':

    np.random.seed(1301) # seed to shuffle the train set

    n_folds = 5
    verbose = True
    shuffle = False

    training = pd.read_csv("../input/train.csv", index_col=0)
    test = pd.read_csv("../input/test.csv", index_col=0)
    print(training.shape)
    print(test.shape)

    # Replace -999999 in var3 column with most common value 2 
    # See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
    # for details
    training = training.replace(-999999,2)

    X = training.iloc[:,:-1]
    y = training.TARGET

    # remove constant columns
    remove = []
    for col in X.columns:
        if X[col].std() == 0:
            remove.append(col)

    X.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    selectK = SelectKBest(f_classif, k=220)
    selectK.fit(X, y)
    X_sel = selectK.transform(X)

    features = X.columns[selectK.get_support()]
    print (features)
    sel_test = selectK.transform(test) 
    X, y, X_submission = np.array(X_sel), np.array(y.astype(int)).ravel(), np.array(sel_test)

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini', class_weight='balanced'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy', class_weight='balanced'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1, criterion='gini', class_weight='balanced_subsample', bootstrap=True, oob_score=True,
   min_samples_leaf=3, max_features='log2', max_depth=5),
#            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
            xgb.XGBClassifier(max_depth = 5,
                n_estimators=525,
                learning_rate=0.05, 
                nthread=4,
                subsample=0.95,
                colsample_bytree=0.85, 
                seed=4242)]

    print ("Creating train and test sets for blending.")
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    skf = cross_validation.StratifiedKFold(y, n_folds, shuffle=True)
    
    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
        for i, (train, testidx) in enumerate(skf):
            print ("Fold", i)
            X_train, y_train = X[train], y[train]
            X_test, y_test = X[testidx], y[testidx]
#            clf.fit(X_train, y_train)
            if j < len(clfs)-1:
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
                    eval_set=[(X_test, y_test)])
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[testidx, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(axis=1)

    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    submission = pd.DataFrame({"ID":test.index, "TARGET":y_submission})
    submission.to_csv("submission_2xRF2xETGB004.csv", index=False)
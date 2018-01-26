import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


def rmsle(preds,actuals):
	return np.sum((np.log(np.array(preds)+1) - np.log(np.array(actuals)+1))**2 / len(preds))
	
def load_data(path):
    df = pd.read_csv(path, parse_dates='quote_date')
    return df
	
def make_submission(clf, X_test, ids, encoder, name='rf_calibrated1.csv'):
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.4f')
	print("Wrote submission to file {}.".format(name))

train = load_data('train_set.csv')
test = load_data('test_set.csv')

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1)

# Train random forest classifier
rf_model = RandomForestClassifier(n_estimators=500, max_features=32)

rf_model.fit(X_train, y_train)
rf_model.pred = rf_model.predict(X_val)
score = log_loss(y_val, rf_model.pred)
print(score)

make_submission(sig_clf, X_test, ids, encoder)

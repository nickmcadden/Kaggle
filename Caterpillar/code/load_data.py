import pandas as pd
import numpy as np
from sklearn import ensemble, preprocessing

def load():
    train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
    test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
    tube_data = pd.read_csv('../input/tube.csv')
    bill_of_materials_data = pd.read_csv('../input/bill_of_materials.csv')
    specs_data = pd.read_csv('../input/specs.csv')

    train = pd.merge(train, tube_data, on ='tube_assembly_id')
    train = pd.merge(train, bill_of_materials_data, on ='tube_assembly_id')
    test = pd.merge(test, tube_data, on ='tube_assembly_id')
    test = pd.merge(test, bill_of_materials_data, on ='tube_assembly_id')

    idx = test.id.values.astype(int)

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month

    # drop useless columns and create labels
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
    labels = train.cost.values
    labels = labels.astype(theano.config.floatX)

    #for some reason material_id cannot be converted to categorical variable
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

    train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    test['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        # print(column_label)
        train[column_label].replace(np.nan,' ', regex=True, inplace= True)
        test[column_label].replace(np.nan,' ', regex=True, inplace= True)

    train.fillna(0, inplace = True)
    test.fillna(0, inplace = True)

    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)

    # label encode the categorical variables
    for i in range(train.shape[1]):
        if i in [0,3,5,11,12,13,14,15,16,20,22,24,26,28,30,32,34]:
            print(i,list(train[1:5,i]) + list(test[1:5,i]))
            lbl = LabelEncoder()
            lbl.fit(list(train[:,i]) + list(test[:,i]))
            train[:,i] = lbl.transform(train[:,i])
            test[:,i] = lbl.transform(test[:,i])
	
    train = train.astype(theano.config.floatX)
    test = test.astype(theano.config.floatX)
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, labels, test, idx
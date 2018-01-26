import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
from sklearn.utils import shuffle

def get_sparse_specs():
    train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
    test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
    tube = pd.read_csv('../input/tube.csv', index_col=0, true_values=['Y'],false_values=['N'])
    materials = pd.read_csv('../input/bill_of_materials.csv', index_col=0)
    specs = pd.read_csv('../input/specs.csv', index_col=0)
    components = pd.read_csv('../input/components.csv', index_col=0)
    aggs = pd.read_csv('../input/ta_aggs.csv')

    train = pd.merge(train, aggs, on ='tube_assembly_id', how='left')
    test = pd.merge(test, aggs, on ='tube_assembly_id', how='left')

    '''
    print('encoding tube components')
    for idx in tube.index:
        for field in ['material_id', 'end_a', 'end_x']:
            if tube.ix[idx, field] is not np.nan:
                tube.ix[idx, tube.ix[idx, field]] = 1
        for i in range(1, 9):
            if materials.ix[idx, 'component_id_%d' % i] is not np.nan:
                #tube.ix[idx, materials.ix[idx, 'component_id_%d' % i]] = materials.ix[idx, 'quantity_%d' % i]
                tube.ix[idx, components.ix[materials.ix[idx, 'component_id_%d' % i], 'component_type_id']] = materials.ix[idx, 'quantity_%d' % i]
        #for i in range(1, 11):
            #if specs.ix[idx, 'spec%d' % i] is not np.nan:
                #tube.ix[idx, specs.ix[idx, 'spec%d' % i]] = 1
    print(tube.shape)

    tube[['end_a_1x','end_a_2x','end_x_1x','end_x_2x']] = tube[['end_a_1x','end_a_2x','end_x_1x','end_x_2x']].astype(int)
    tube.to_csv('../input/tube_semi_sparse.csv')
    '''
    tube = pd.read_csv('../input/tube_semi_sparse.csv')

    print('sorting tube columns')
    sparse_cols = tube.columns.tolist()
    sparse_cols.sort()
    tube = tube[sparse_cols]

    print('labelling suppliers')
    lbl = LabelEncoder()
    lbl.fit(list(train.ix[:,'supplier']) + list(test.ix[:,'supplier']))
    train.ix[:,'supplier'] = lbl.transform(train.ix[:,'supplier'])
    test.ix[:,'supplier'] = lbl.transform(test.ix[:,'supplier'])

    print('encoding train suppliers')
    for idx in train.index:
	    if train.ix[idx, 'supplier'] is not np.nan:
		    train.ix[idx, 'supplier_%d' % train.ix[idx, 'supplier']] = 1
    print(train.shape)
    train.to_csv('../input/train_sparse.csv')

    print('encoding test suppliers')
    for idx in test.index:
	    if test.ix[idx, 'supplier'] is not np.nan:
		    test.ix[idx, 'supplier_%d' % test.ix[idx, 'supplier']] = 1
    print(test.shape)
    test.to_csv('../input/test_sparse.csv')

    train['bracket_pricing'] = train['bracket_pricing'].replace(['Yes','No'],[1,0])
    test['bracket_pricing'] = test['bracket_pricing'].replace(['Yes','No'],[1,0])
	
    train = pd.merge(train, tube, on ='tube_assembly_id')
    test = pd.merge(test, tube, on ='tube_assembly_id')

    train.drop(['supplier', 'material_id', 'end_a', 'end_x'], inplace=True, axis=1)
    test.drop(['supplier', 'material_id', 'end_a', 'end_x'], inplace=True, axis=1)

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month

    train.to_csv('../input/train_sparse.csv', index=False)
    test.to_csv('../input/test_sparse.csv', index=False)

    return train, test

def get_dense_specs():
    train = pd.read_csv('../input/train_set.csv', parse_dates=[2,])
    test = pd.read_csv('../input/test_set.csv', parse_dates=[3,])
    tube = pd.read_csv('../input/tube.csv', true_values=['Y'],false_values=['N'])
    materials = pd.read_csv('../input/bill_of_materials.csv')
    aggs = pd.read_csv('../input/ta_aggs.csv')
    components = pd.read_csv('../input/components.csv')

    train = pd.merge(train, tube, on ='tube_assembly_id')
    test = pd.merge(test, tube, on ='tube_assembly_id')
    train = pd.merge(train, materials, on ='tube_assembly_id')
    test = pd.merge(test, materials, on ='tube_assembly_id')

    train = pd.merge(train, aggs, on ='tube_assembly_id', how='left')
    test = pd.merge(test, aggs, on ='tube_assembly_id', how='left')

    # create some new features
    train['year'] = train.quote_date.dt.year
    train['month'] = train.quote_date.dt.month

    test['year'] = test.quote_date.dt.year
    test['month'] = test.quote_date.dt.month

    train['odd'] = train.quantity % 2
    test['odd'] = test.quantity % 2

    train['div5'] = (train.quantity % 5)
    test['div5'] = (test.quantity % 5)

    train['material_id'].replace(np.nan,' ', regex=True, inplace= True)
    test['material_id'].replace(np.nan,' ', regex=True, inplace= True)

    train['bracket_pricing'] = train['bracket_pricing'].replace(['Yes','No'],[1,0])
    test['bracket_pricing'] = test['bracket_pricing'].replace(['Yes','No'],[1,0])

    fields_to_encode = ['supplier', 'material_id', 'end_a', 'end_x', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'bracket_pricing']
 
    for i in range(1,9):
        column_label = 'component_id_'+str(i)
        fields_to_encode.append(column_label)
        tmp = pd.merge(train, components, left_on =column_label, right_on='component_id', how='left')['component_type_id']
        train[column_label] = tmp
        tmp = pd.merge(test, components, left_on =column_label, right_on='component_id', how='left')['component_type_id']
        test[column_label] = tmp
        train[column_label].replace(np.nan,' ', regex=True, inplace= True)
        test[column_label].replace(np.nan,' ', regex=True, inplace= True)

    for j, clf in enumerate(train.columns.tolist()):
        print(j, clf)
    '''    
    # label encode the categorical variables
    for i in fields_to_encode:
        print('Encoding',i)
        lbl = LabelEncoder()
        lbl.fit(list(train.ix[:,i]) + list(test.ix[:,i]))
        train.ix[:,i] = lbl.transform(train.ix[:,i])
        test.ix[:,i] = lbl.transform(test.ix[:,i])

    for i in fields_to_encode:
        print('Encoding',i)
        freqs = train[i].append(test[i]).value_counts()
        train[i] = pd.match(train[i].values, freqs[0:45].index)
        test[i] = pd.match(test[i].values, freqs[0:45].index)
    '''
    for i in fields_to_encode:
        print('Encoding',i)
        rank = pd.concat([train[i],train['cost']], axis=1).groupby(i).mean().sort('cost', ascending=False)
        print(rank[0:20])
        train[i] = pd.match(train[i].values, rank[0:45].index)
        test[i] = pd.match(test[i].values, rank[0:45].index)

    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)

    return train, test

class KLabelFolds():
  def __init__(self, labels, n_folds=3):
    self.labels = labels
    self.n_folds = n_folds

  def __iter__(self):
    unique_labels = self.labels.unique()
    cv = cross_validation.KFold(len(unique_labels), self.n_folds, shuffle=True)
    for train, test in cv:
      test_labels = unique_labels[test]
      test_mask = self.labels.isin(test_labels)
      train_mask = np.logical_not(test_mask)
      yield (np.where(train_mask)[0], np.where(test_mask)[0])

def load(sparse=True):

    if sparse:
        #train = pd.read_csv('../input/train_sparse.csv')
        #test = pd.read_csv('../input/test_sparse.csv')
        train, test = get_sparse_specs()
        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)
    else:
        train, test = get_dense_specs()
    
    idx = test.id.values.astype(int)
    labels = train.cost.values

    cv = list(KLabelFolds(train.tube_assembly_id, 4))

    # drop useless columns and create labels
    test = test.drop(['id', 'tube_assembly_id', 'quote_date'], axis = 1)
    train = train.drop(['quote_date', 'cost', 'tube_assembly_id'], axis = 1)

    print("train.shape", train.shape, test.shape)

    q=[]
    for i in range(0,train.shape[1]):
        if (train.ix[:,i]!=0).sum() < 10:
            q.append(train.columns[i])
    for i in range(0,test.shape[1]):
        if (test.ix[:,i]!=0).sum() < 10:
            q.append(test.columns[i])
    q=list(set(q))
    print("dropping", [val for val in q if val in train.columns])
    train = train.drop([val for val in q if val in train.columns.values.tolist()], axis = 1)
    test = test.drop([val for val in q if val in test.columns.values.tolist()], axis = 1)
    print(pd.DataFrame({'a':train.columns.values,'b':test.columns.values, 'c':(train!=0).sum(axis=0)}))

    cols = train.columns.tolist()

    # convert data to numpy array
    train = np.array(train)
    test = np.array(test)

    print('transforming data')
    #scaler = StandardScaler()
    #train = scaler.fit_transform(train)
    #test = scaler.fit_transform(test)

    return train, labels, test, idx, cols, cv
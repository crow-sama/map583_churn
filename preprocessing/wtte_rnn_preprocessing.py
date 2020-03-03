import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


def get_CMAPSSData(nb_file):
    # get data from file and pre process it (normalization and convert to pandas)
    dataset_train = pd.read_csv('RUL-Net/CMAPSSData/train_FD00{}.txt'.format(nb_file),
                                sep=' ', header=None).drop([26, 27], axis=1)
    dataset_test = pd.read_csv('RUL-Net/CMAPSSData/test_FD00{}.txt'.format(nb_file),
                               sep=' ', header=None).drop([26, 27], axis=1)
    test_truth = pd.read_csv('RUL-Net/CMAPSSData/RUL_FD00{}.txt'.format(nb_file),
                             sep=' ', header=None).drop([1], axis=1)
    col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                 's9',
                 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    dataset_train.columns = col_names
    dataset_test.columns = col_names
    test_truth.columns = ['more']
    test_truth['id'] = test_truth.index + 1
    rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    test_truth['rtf'] = test_truth['more'] + rul['max']
    test_truth.drop('more', axis=1, inplace=True)
    dataset_test = dataset_test.merge(test_truth, on=['id'], how='left')
    dataset_test['ttf'] = dataset_test['rtf'] - dataset_test['cycle']
    dataset_test.drop('rtf', axis=1, inplace=True)
    dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max) - dataset_train['cycle']
    features_col_name = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
                         's9', 's10', 's11',
                         's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    target_col_name = 'ttf'
    relevant_features_col_name = []
    for col in features_col_name:
        if not (len(dataset_train[col].unique()) == 1):
            relevant_features_col_name.append(col)
    sc = MinMaxScaler()
    dataset_train[features_col_name] = sc.fit_transform(dataset_train[features_col_name])
    dataset_test[features_col_name] = sc.transform(dataset_test[features_col_name])
    return dataset_train, dataset_test, relevant_features_col_name, target_col_name


def to_lists_of_tensors(dataset, features_col_name, target_col_name):
    # take pandas df and convert it to list of tensors (for pytorch)
    X, y = [], []
    nb_sequences = max(dataset['id'])
    for i in range(1, nb_sequences + 1):
        df_zeros = dataset.loc[dataset['id'] == i]
        df_one_x = df_zeros[features_col_name]
        df_one_y = df_zeros[target_col_name]
        X.append(torch.from_numpy(np.expand_dims(df_one_x.values, 1)).type(torch.FloatTensor))
        y.append(torch.from_numpy(df_one_y.values).type(torch.FloatTensor))
    return X, y


def convert_train_and_test_to_appropriate_format(dataset_train, dataset_test, features_col_name, target_col_name):
    # take 2 datasets (train and test and covert them to lists of tensors)
    X_train, y_train = to_lists_of_tensors(dataset_train, features_col_name, target_col_name)
    X_test, y_test = to_lists_of_tensors(dataset_test, features_col_name, target_col_name)
    return X_train, y_train, X_test, y_test

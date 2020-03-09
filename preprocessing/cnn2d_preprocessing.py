import pandas as pd

import torch

from sklearn.preprocessing import MinMaxScaler
from skearn.feature_selection import VarianceThreshold
from sklearn.spatial.distance import pdist, squareform


def read_raw_data(path):
    """
    Read data (or test) files and add a `ttf` column.
    """
    data = pd.read_csv(path, sep=" ", header=None)

    # drop columns containing NaN(s)
    data.drop([26,27], axis=1, inplace=True)

    col_names = [
        "id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
    ]
    data.columns = col_names

    # time to fail
    data["ttf"] =  data.groupby("id")["cycle"].transform(max)- data["cycle"]

    return data


def read_test_data(test_path, truth_path):
    """
    Read test file and adjust the `ttf` column with truth data (remaining time to fail).
    """
    base = read_raw_data(test_path)

    truth = pd.read_csv(truth_path, sep=" ", header=None)

    # drop column containing NaN, sync id
    truth = truth.drop([1], axis=1)
    truth.columns = ["more"]
    truth["id"] = truth.index + 1

    # add remaining cycles
    data = base.merge(truth, on=["id"], how="left")
    data["ttf"] += data["more"]

    return data


def read_data(file_number):
    train_path = "RUL-Net/CMAPSSData/train_FD00{}.txt".format(file_number)
    test_path = "RUL-Net/CMAPSSData/test_FD00{}.txt".format(file_number)
    truth_path = "RUL-Net/CMAPSSData/RUL_FD00{}.txt".format(file_number)

    train = read_raw_data(train_path)
    test = read_test_data(test_path, truth_path) 

    features = [
        "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"
    ]
    
    # scale features
    scaler = MinMaxScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

    # remove low variance features
    var_filter = VarianceThreshold()
    var_filter.fit(train[features])
    mask = var_filter.get_support()
    selected_features = [
        features[i] for i, flag in enumerate(mask) if flag
    ]

    # create a three-class label (label2)
    w1 = 45
    w0 = 15

    train["label1"] = np.where(train["ttf"] <= w1, 1, 0)
    train["label2"] = train["label1"]
    train.loc[train["ttf"] <= w0, "label2"] = 2
    
    test["label1"] = np.where(test["ttf"] <= w1, 1, 0)
    test["label2"] = test["label1"]
    test.loc[test["ttf"] <= w0, "label2"] = 2

    return train, test, selected_features


def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # iterate with (i, i + seq_length)
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # for each sequence keep the last label
    return data_matrix[seq_length:num_elements, :]


def gen_features(df, sequence_length, sequence_cols):
    feature_data = []
    for engine_id in df.id.unique():
        for sequence in gen_sequence(df[df.id == engine_id], sequence_length, sequence_cols):
            feature_data.append(sequence)
    return np.asarray(feature_data)


def gen_responses(df, sequence_length, label_cols):
    responses = []
    for engine_id in df.id.unique():
        for label in gen_labels(df[df.id == engine_id], sequence_length, label_cols):
            responses.append(label)
    return np.asarray(responses).reshape(-1, 1)


def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


class Loader:

    def __init__(self, use_cuda, dataset_number, batch_size):

        train_df, test_df, feature_cols = read_data(dataset_number)

        X_train = gen_features(
            df=train_df,
            sequence_length=50,
            sequence_cols=feature_cols,
        )

        X_test = gen_features(
            df=test_df,
            sequence_length=50,
            sequence_cols=feature_cols,
        )

        self.y_train = gen_responses(
            df=train_df,
            sequence_length=50,
            label_cols=["label2"],
        )

        self.y_test = gen_responses(
            df=test_df,
            sequence_length=50,
            label_cols=["label2"],
        )

        self.X_train = np.apply_along_axis(rec_plot, 1, X_train).astype('float16')

        self.X_test = np.apply_along_axis(rec_plot, 1, X_test).astype('float16')

        self.num_channels = len(feature_cols)

        self.num_train_batches = int(X_train.shape[0] / batch_size)
        self.num_test_batches = int(X_test.shape[0] / batch_size)

        self.cuda = use_cuda and torch.cuda_is_available()

        self.batch_size = batch_size


    def train(self):
        for i in range(self.num_train_batches):
            start, stop = i * self.batch_size, (i + 1) * self.batch_size
            X = self.X_train[start:stop]
            y = self.y_train[start:stop]

            X = torch.from_numpy(X).type(torch.FloatTensor)
            y = torch.from_numpy(y).type(torch.LongTensor)

            if self.cuda:
                X, y = X.cuda(), y.cuda()

            yield X, y


    def test(self):
        for i in range(self.num_test_batches):
            start, stop = i * self.batch_size, (i + 1) * self.batch_size
            X = self.X_test[start:stop]
            y = self.y_test[start:stop]

            X = torch.from_numpy(X).type(torch.FloatTensor)
            y = torch.from_numpy(y).type(torch.LongTensor)

            if self.cuda:
                X, y = X.cuda(), y.cuda()

            yield X, y


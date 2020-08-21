# coding: utf-8
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from config_model import TRAIN_VALIDATION_TEST_FOLDER


def train_validation_test_split(iris_dataset_df):
    shuffled_indices_setosa = np.random.permutation(np.arange(0, 50))
    shuffled_indices_versicolor = np.random.permutation(np.arange(50, 100))
    shuffled_indices_virginica = np.random.permutation(np.arange(100, 150))

    train_indices = np.concatenate([shuffled_indices_setosa[0:40],
                                    shuffled_indices_versicolor[0:40],
                                    shuffled_indices_virginica[0:40]])
    validation_indices = np.concatenate([shuffled_indices_setosa[40:45],
                                         shuffled_indices_versicolor[40:45],
                                         shuffled_indices_virginica[40:45]])
    test_indices = np.concatenate([shuffled_indices_setosa[45:50],
                                   shuffled_indices_versicolor[45:50],
                                   shuffled_indices_virginica[45:50]])

    train_set = iris_dataset_df.loc[train_indices]
    validation_set = iris_dataset_df.loc[validation_indices]
    test_set = iris_dataset_df.loc[test_indices]
    train_set.to_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "train_set.csv"), sep=";", index=False)
    validation_set.to_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "validation_set.csv"), sep=";", index=False)
    test_set.to_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "test_set.csv"), sep=";", index=False)

def X_y_extraction(iris_df):
    # one-hot encoding
    iris_df = pd.get_dummies(iris_df, columns=['Species'])
    # (X, y) split
    col_names = iris_df.columns
    X = iris_df[col_names[1:-3]]
    X = np.array(X, dtype='float32')
    y = iris_df[col_names[-3:]]
    y = np.array(y, dtype='float32')
    return X, y

def make_scaler(X, scaler_file):
    X_scaler = StandardScaler()
    # Here all the 4 columns need to be scaled
    X_scaler.fit(X)
    joblib.dump(X_scaler, scaler_file)

def scale(X, scaler_file):
    X_scaler = joblib.load(scaler_file)
    X_scaled = X_scaler.transform(X)
    return X_scaled
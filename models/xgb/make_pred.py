# coding: utf-8
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from config_model import IRIS_DATASET_FILE, MODEL_FOLDER


if __name__ == '__main__':

    print("Starting the training of a decision tree")

    # Loading data
    iris_dataset_df = pd.read_csv(IRIS_DATASET_FILE)
    iris_dataset_df = pd.get_dummies(iris_dataset_df, columns=['Species'])
    col_names = iris_dataset_df.columns
    X = iris_dataset_df[col_names[1:-3]]
    X = np.array(X, dtype='float32')
    y = iris_dataset_df[col_names[-3:]]
    y = np.array(y, dtype='float32')
    y = np.argmax(y, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # No scaling for a tree-based model
    model_name = "xgb"
    trained_model_folder = os.path.join(MODEL_FOLDER, "trained_model")

    model = joblib.load(os.path.join(trained_model_folder, model_name + ".save"))

    y_pred_decision_tree = model.predict(X_test)

    print(y_pred_decision_tree)
    print(y_test)

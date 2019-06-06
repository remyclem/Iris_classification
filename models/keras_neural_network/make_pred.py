# coding: utf-8
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from config_model import TRAIN_VALIDATION_TEST_FOLDER, MODEL_FOLDER, TRAINED_MODEL_FOLDER, \
                         X_SCALER_FILE, HDF5_MODEL
from data_processing import X_y_extraction, scale


def make_pred(X,
              scaler_file,
              model_hdf5):
    X_scaled = scale(X, scaler_file)

    # load weights into new model
    model = load_model(model_hdf5)
    print("Keras model restored.")

    y_pred = model.predict(X_scaled)

    return y_pred


if __name__ == '__main__':

    print("making predictions")
    test_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "test_set.csv"), sep=";")
    X_test, y_test = X_y_extraction(test_set_df)

    y_pred = make_pred(X_test,
                       X_SCALER_FILE,
                       HDF5_MODEL)
    y_pred_reformated = np.argmax(y_pred, axis=1)

    # Lets check
    y_test_reformated = np.argmax(y_test, axis=1)  # Because of one-hot encoding
    print(y_pred_reformated)
    print(y_test_reformated)
# coding: utf-8
import os

MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__))
TRAIN_VALIDATION_TEST_FOLDER = os.path.join(MODEL_FOLDER, "train_validation_test_sets")
IRIS_DATASET_FILE = os.path.join(MODEL_FOLDER, "..", "..", "data", "Iris_dataset.csv")
TRAINED_MODEL_FOLDER = os.path.join(MODEL_FOLDER, "trained_model")
HDF5_MODEL = os.path.join(TRAINED_MODEL_FOLDER, "keras_neural_network" + ".h5")
SCORE_MODEL = os.path.join(TRAINED_MODEL_FOLDER, "eval_keras_neural_network" + ".csv")
X_SCALER_FOLDER = os.path.join(TRAINED_MODEL_FOLDER, "scalers")
X_SCALER_FILE = os.path.join(X_SCALER_FOLDER, "X_scaler.save")


if __name__ == '__main__':
    print("MODEL_FOLDER: " + MODEL_FOLDER)
    print("TRAIN_VALIDATION_TEST_FOLDER: " + TRAIN_VALIDATION_TEST_FOLDER)
    print("IRIS_DATASET_FILE: " + IRIS_DATASET_FILE)
    print("TRAINED_MODEL_FOLDER: " + TRAINED_MODEL_FOLDER)
    print("X_SCALER_FOLDER: " + X_SCALER_FOLDER)
    print("X_SCALER_FILE: " + X_SCALER_FILE)
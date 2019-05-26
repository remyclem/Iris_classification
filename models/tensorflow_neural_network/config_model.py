# coding: utf-8
import os

MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__))
TRAIN_VALIDATION_TEST_FOLDER = os.path.join(MODEL_FOLDER, "train_validation_test_sets")
IRIS_DATASET_FILE = os.path.join(MODEL_FOLDER, "..", "..", "data", "Iris_dataset.csv")

if __name__ == '__main__':
    print("MODEL_FOLDER: " + MODEL_FOLDER)
    print("TRAIN_VALIDATION_TEST_FOLDER: " + TRAIN_VALIDATION_TEST_FOLDER)
    print("IRIS_DATASET_FILE: " + IRIS_DATASET_FILE)
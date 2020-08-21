# coding: utf-8
import os

MODEL_FOLDER = os.path.dirname(os.path.abspath(__file__))
IRIS_DATASET_FILE = os.path.join(MODEL_FOLDER, "..", "..", "data", "iris_dataset.csv")


if __name__ == '__main__':
    print("MODEL_FOLDER: " + MODEL_FOLDER)
    print("IRIS_DATASET_FILE: " + IRIS_DATASET_FILE)
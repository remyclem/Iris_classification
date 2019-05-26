# coding: utf-8
import os

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
MODELS_FOLDER = os.path.join(ROOT_FOLDER, "models")
IRIS_DATASET_FILE = os.path.join(DATA_FOLDER, "Iris_dataset.csv")


if __name__ == '__main__':
    print("ROOT_FOLDER: " + ROOT_FOLDER)
    print("DATA_FOLDER: " + DATA_FOLDER)
    print("MODELS_FOLDER: " + MODELS_FOLDER)
    print("IRIS_DATASET_FILE: " + IRIS_DATASET_FILE)
# coding: utf-8
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import datetime
from config_model import TRAIN_VALIDATION_TEST_FOLDER, MODEL_FOLDER, IRIS_DATASET_FILE, \
                         TRAINED_MODEL_FOLDER, X_SCALER_FOLDER, X_SCALER_FILE, HDF5_MODEL
from data_processing import X_y_extraction, make_scaler, scale, train_validation_test_split


def train_model(X_train, y_train,
                X_validation, y_validation,
                save_file,
                tensorboard_log_folder,
                learning_rate=0.001,
                nb_epochs=500,
                minibatch_size=16):

    nb_input_neurons = X_train.shape[1]
    nb_hidden_layer_neurons = nb_input_neurons
    nb_output_neurons = y_train.shape[1]

    model = Sequential()
    model.add(Dense(nb_hidden_layer_neurons, input_dim=nb_input_neurons, activation='relu'))
    model.add(Dense(nb_hidden_layer_neurons, activation='relu'))
    model.add(Dense(nb_output_neurons, activation='softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # For tensorboard feedback, open a terminal and write:
    # tensorboard --logdir=models/keras_neural_network/tensorboard_log_folder
    tensorboard = TensorBoard(log_dir=tensorboard_log_folder)

    model.fit(X_train, y_train,
              validation_data=(X_validation, y_validation),
              epochs=nb_epochs,
              batch_size=minibatch_size,
              callbacks=[tensorboard],
              verbose=0)

    train_results = model.evaluate(X_train, y_train)  # [loss, metric=accuracy]
    validation_results = model.evaluate(X_validation, y_validation)
    print(train_results)
    print(validation_results)

    model.save(save_file)

    return model


if __name__ == '__main__':

    print("Starting the training of the neural network")

    # Settings
    now = datetime.datetime.now()
    tensorboard_log_folder = os.path.join(TRAINED_MODEL_FOLDER, "tensorboard_log_folder",
                                          "run-" + datetime.datetime.strftime(now, format="%Y_%m_%d_%H%M"))
    if not os.path.isdir(TRAINED_MODEL_FOLDER):
        os.makedirs(TRAINED_MODEL_FOLDER)
    if not os.path.isdir(tensorboard_log_folder):
        os.makedirs(tensorboard_log_folder)
    if not os.path.isdir(X_SCALER_FOLDER):
        os.makedirs(X_SCALER_FOLDER)

    # Splitting the initial dataset into train-validation-test
    make_train_validation_test_split = True
    if make_train_validation_test_split:
        iris_dataset_df = pd.read_csv(IRIS_DATASET_FILE)
        train_validation_test_split(iris_dataset_df)
    train_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "train_set.csv"), sep=";")
    X_train, y_train = X_y_extraction(train_set_df)
    validation_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "validation_set.csv"), sep=";")
    X_validation, y_validation = X_y_extraction(validation_set_df)

    # Scaling the input data
    make_scaler(X_train, X_SCALER_FILE)
    X_train_scaled = scale(X_train, X_SCALER_FILE)
    X_validation_scaled = scale(X_validation, X_SCALER_FILE)

    # Training
    trained_model = train_model(X_train_scaled, y_train,
                                X_validation_scaled, y_validation,
                                HDF5_MODEL,
                                tensorboard_log_folder,
                                learning_rate=0.01,
                                nb_epochs=500,
                                minibatch_size=8)

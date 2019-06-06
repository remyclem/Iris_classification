# coding: utf-8
import os
import tempfile
import pandas as pd
import datetime
from config_model import TRAIN_VALIDATION_TEST_FOLDER, IRIS_DATASET_FILE
from data_processing import X_y_extraction, make_scaler, scale, train_validation_test_split
from train_model import train_model

# Normally a random search would be more efficient, but here I can
# afford to be lazy

if __name__ == '__main__':

    root_tuning_folder = os.path.join(tempfile.gettempdir(), "iris_hyperparameters_tuning")
    print("hello " + root_tuning_folder)

    # Splitting the initial dataset into train-validation-test
    make_train_validation_test_split = True
    if make_train_validation_test_split:
        iris_dataset_df = pd.read_csv(IRIS_DATASET_FILE)
        train_validation_test_split(iris_dataset_df)
    train_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "train_set.csv"), sep=";")
    X_train, y_train = X_y_extraction(train_set_df)
    validation_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "validation_set.csv"), sep=";")
    X_validation, y_validation = X_y_extraction(validation_set_df)

    # Defining the possible values for hyperparameters
    possible_learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
    possible_nb_additional_layers = [0, 1, 2, 5]

    # Go take a coffee or two
    scores = dict()
    for lr in possible_learning_rates:
        for nb_additional_layers in possible_nb_additional_layers:

            print("Testing - lr={} - nb_additional_layers={}".format(lr, nb_additional_layers))

            # settings
            trained_model_folder = os.path.join(root_tuning_folder,
                                                "trained_model_lr={}_addlayers={}".format(lr, nb_additional_layers))
            if not os.path.isdir(trained_model_folder):
                os.makedirs(trained_model_folder)
            X_scaler_folder = os.path.join(trained_model_folder, "scalers")
            if not os.path.isdir(X_scaler_folder):
                os.makedirs(X_scaler_folder)
            X_scaler_file = os.path.join(X_scaler_folder, "X_scaler.save")
            now = datetime.datetime.now()
            tensorboard_log_folder = os.path.join(trained_model_folder, "tensorboard_log_folder",
                                                  "run-" + datetime.datetime.strftime(now, format="%Y_%m_%d_%H%M"))
            if not os.path.isdir(tensorboard_log_folder):
                os.makedirs(tensorboard_log_folder)
            hdf5_model = os.path.join(trained_model_folder, "keras_neural_network" + ".h5")
            score_model = os.path.join(trained_model_folder, "eval_keras_neural_network" + ".csv")

            # scaling and saving the scaler at the right location
            make_scaler(X_train, X_scaler_file)
            X_train_scaled = scale(X_train, X_scaler_file)
            X_validation_scaled = scale(X_validation, X_scaler_file)

            # training a model with the selected hyperparameters
            trained_model = train_model(X_train_scaled, y_train,
                                        X_validation_scaled, y_validation,
                                        hdf5_model,
                                        score_model,
                                        tensorboard_log_folder,
                                        nb_additional_layers=nb_additional_layers,
                                        learning_rate=lr,
                                        nb_epochs=500,
                                        minibatch_size=8)

            eval_df = pd.read_csv(score_model, sep=";", index_col=[0])
            val_accuracy = eval_df.loc["validation"]["accuracy"]
            current_hyperparameters = "lr={}_addlayers={}".format(lr, nb_additional_layers)
            scores[current_hyperparameters] = val_accuracy

    # Displaying the scores
    print(sorted(scores.items(), reverse=True, key=lambda kv: (kv[1], kv[0])))
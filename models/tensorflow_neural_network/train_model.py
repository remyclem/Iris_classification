# coding: utf-8
import os
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import datetime
from config_model import TRAIN_VALIDATION_TEST_FOLDER, MODEL_FOLDER, IRIS_DATASET_FILE
from data_processing import X_y_extraction, make_scaler, scale, train_validation_test_split
from utils import random_mini_batches

# TODO: improve tensorboard
def train_model(X_train, y_train,
                X_validation, y_validation,
                model_name,
                tf_logs,
                trained_model_folder,
                learning_rate=0.01,
                nb_epochs=500,
                minibatch_size=16):

    nb_input_neurons = X_train.shape[1]
    nb_hidden_layer_neurons = nb_input_neurons
    nb_output_neurons = y_train.shape[1]

    X_placeholder = tf.placeholder(shape=[None, nb_input_neurons], dtype=tf.float32, name="X_placeholder")
    y_placeholder = tf.placeholder(shape=[None, nb_output_neurons], dtype=tf.float32, name="y_placeholder")

    with tf.name_scope("dnn"):
        # just one hidden layer is enough here
        hidden = tf.layers.dense(X_placeholder,
                                   nb_hidden_layer_neurons,
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.zeros_initializer(),
                                   name="hidden")
        # layer 2 - logits = wx+b = z, final_output = activation(z)
        logits = tf.layers.dense(hidden, nb_output_neurons, name="logits")
        probas = tf.nn.softmax(logits, name="probas")

    with tf.name_scope("loss"):
        x_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_placeholder)
        loss = tf.reduce_mean(x_entropy, name="loss")

    with tf.name_scope("accuracy"):
        labels = tf.argmax(y_placeholder, 1)
        predictions = tf.argmax(logits, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    init_global = tf.global_variables_initializer()

    saver = tf.train.Saver()

    loss_summary = tf.summary.scalar("loss", loss)
    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    validation_summary = tf.summary.scalar("validation_accuracy", accuracy)
    summary_writer = tf.summary.FileWriter(tf_logs, tf.get_default_graph())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(init_global)

        for epoch in tqdm(range(nb_epochs), desc="epoch"):

            minibatches = random_mini_batches(X_train, y_train, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_y) = minibatch
                sess.run(training_op, feed_dict={X_placeholder: minibatch_X, y_placeholder: minibatch_y})

            if epoch % 10 == 0:
                # For tensorboard feedback, open a terminal and write:
                # tensorboard --logdir=models/tensorflow_neural_network/tf_logs
                summary_loss = loss_summary.eval(feed_dict={X_placeholder: X_train, y_placeholder: y_train})
                summary_train_acc = sess.run(training_summary, feed_dict={X_placeholder: X_train, y_placeholder: y_train})
                summary_valid_acc = sess.run(validation_summary, feed_dict={X_placeholder: X_validation, y_placeholder: y_validation})
                summary_writer.add_summary(summary_loss, epoch)
                summary_writer.add_summary(summary_train_acc, epoch)
                summary_writer.add_summary(summary_valid_acc, epoch)

            if epoch % 100 == 0 and epoch > 200:
                epoch_folder = os.path.join(trained_model_folder, "tmp", str(epoch))
                if not os.path.isdir(epoch_folder):
                    os.makedirs(epoch_folder)
                tmp_name = "neural_net_epoch_{}.ckpt".format(str(epoch))
                saver.save(sess, os.path.join(epoch_folder, tmp_name))

        saver.save(sess, os.path.join(trained_model_folder, "{}.ckpt".format(model_name)))
        summary_writer.close()


if __name__ == '__main__':

    print("Starting the training of the neural network")

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
    X_scaler_file = os.path.join(MODEL_FOLDER, "scalers", "X_scaler.save")
    make_scaler(X_train, X_scaler_file)
    X_train_scaled = scale(X_train, X_scaler_file)
    X_validation_scaled = scale(X_validation, X_scaler_file)

    # Training
    model_name = "tensorflow_neural_network"
    now = datetime.datetime.now()
    tf_logs = os.path.join(MODEL_FOLDER, "tf_logs",
                           "run-" + datetime.datetime.strftime(now, format="%Y_%m_%d_%H%M"))
    trained_model_folder = os.path.join(MODEL_FOLDER, "trained_model")
    train_model(X_train_scaled, y_train,
                X_validation_scaled, y_validation,
                model_name,
                tf_logs,
                trained_model_folder,
                learning_rate=0.01,
                nb_epochs=1000,
                minibatch_size=16)

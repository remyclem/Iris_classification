# coding: utf-8
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from config_model import TRAIN_VALIDATION_TEST_FOLDER, MODEL_FOLDER, TRAINED_MODEL_FOLDER, X_SCALER_FILE, MODEL_NAME
from data_processing import X_y_extraction, scale


def make_pred(X,
              scaler_file,
              trained_model_folder,
              meta_graph):
    X_scaled = scale(X, scaler_file)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(sess, tf.train.latest_checkpoint(trained_model_folder))
        print("TF model restored.")

        graph = tf.get_default_graph()

        X_placeholder = graph.get_tensor_by_name("X_placeholder:0")
        probas = graph.get_tensor_by_name("dnn/probas:0")

        y_pred = sess.run(probas, feed_dict={X_placeholder: X_scaled})
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred


if __name__ == '__main__':

    print("making predictions")
    test_set_df = pd.read_csv(os.path.join(TRAIN_VALIDATION_TEST_FOLDER, "test_set.csv"), sep=";")
    X_test, y_test = X_y_extraction(test_set_df)

    meta_graph = os.path.join(TRAINED_MODEL_FOLDER, MODEL_NAME + ".ckpt.meta")
    y_pred = make_pred(X_test,
                       X_SCALER_FILE,
                       TRAINED_MODEL_FOLDER,
                       meta_graph)

    # Lets check
    y_test_reformated = np.argmax(y_test, axis=1)  # Because of one-hot encoding
    print(y_pred)
    print(y_test_reformated)
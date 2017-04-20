from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os


def prepare_data(data_file_name):
    """
    Responsible for cleaning the data file provided from the UCI machine
    learning repository here: http://archive.ics.uci.edu/ml/datasets/Mushroom.
    The function then produces two CSV files appropriately formatted to be
    used in TensorFlow where the CSV files split with respect to
    training and testing data.
    """

    # The header is formed from the 'agaricus-lepiota.name' file found on
    # http://archive.ics.uci.edu/ml/datasets/Mushroom
    header = ['class', 'cap_shape', 'cap_surface',
              'cap_color', 'bruises', 'odor', 'gill_attachment',
              'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
              'stalk_root', 'stalk_surface_above_ring',
              'stalk_surface_below_ring', 'stalk_color_above_ring',
              'stalk_color_below_ring', 'veil_type', 'veil_color',
              'ring_number', 'ring_type', 'spore_print_color',
              'population', 'habitat']
    df = pd.read_csv(data_file_name, sep=',', names=header)

    # Entries with a '?' indicate a missing piece of data, and
    # these entries are dropped from our dataset.
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # The class of poisonous or edible is indicated in the data as
    # either 'p' or 'e' respectively. We require that this is numeric,
    # and therefore use '0' to indicate poisonous (or not edible) and
    # '1' to indicate edible.
    df['class'].replace('p', 0, inplace=True)
    df['class'].replace('e', 1, inplace=True)

    # Since we are dealing with non-numeric feature data, or in other
    # words, categorical data, we need to replace these with numerical
    # equivalents. Pandas has a nice function called "get_dummies" that
    # performs this task.
    cols_to_transform = header[1:]
    df = pd.get_dummies(df, columns=cols_to_transform)

    # We can now split the data into two separate data frames,
    # one for training, which will constitute the bulk of the
    # data, and one for testing.
    df_train, df_test = train_test_split(df, test_size=0.1)

    # Determine the number of rows and columns in each of the
    # data frames.
    num_train_entries = df_train.shape[0]
    num_train_features = df_train.shape[1] - 1

    num_test_entries = df_test.shape[0]
    num_test_features = df_test.shape[1] - 1

    # The data frames are written as a temporary CSV file, as we still
    # need to modify the header row to include the number of rows and
    # columns present in the training and testing CSV files.
    df_train.to_csv('train_temp.csv', index=False)
    df_test.to_csv('test_temp.csv', index=False)

    # Append onto the header row the information about how many
    # columns and rows are in each file as TensorFlow requires.
    open("mushroom_train.csv", "w").write(str(num_train_entries) +
                                          "," + str(num_train_features) +
                                          "," + open("train_temp.csv").read())

    open("mushroom_test.csv", "w").write(str(num_test_entries) +
                                         "," + str(num_test_features) +
                                         "," + open("test_temp.csv").read())

    # Finally, remove the temporary CSV files that were generated above.
    os.remove("train_temp.csv")
    os.remove("test_temp.csv")


# Define the test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y


# Define the training inputs
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y


# Classify two new mushroom samples. (expect output: [0,1]),
# i.e. poisonous, edible.
def new_samples():
    return np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                      1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                      0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                      0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                      0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,
                      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                      0, 0, 0, 0, 0, 0, 1]], dtype=np.int)


if __name__ == "__main__":

    MUSHROOM_DATA_FILE = "agaricus-lepiota.data"

    # Prepare the mushroom data for TensorFlow by
    # creating two train / test CSV files.
    prepare_data(MUSHROOM_DATA_FILE)

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename='mushroom_train.csv',
        target_dtype=np.int,
        features_dtype=np.int,
        target_column=0)

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename='mushroom_test.csv',
        target_dtype=np.int,
        features_dtype=np.int,
        target_column=0)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=98)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=2,
        model_dir="/tmp/mushroom_model")

    # Fit model.
    classifier.fit(input_fn=get_train_inputs, steps=2000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Test on two mushroom samples.
    predictions = list(classifier.predict(input_fn=new_samples))

    print("New Samples, Class Predictions:    {}\n"
          .format(predictions))

# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

# TensorFlow Dependencies
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

import math

# Allow GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 1
fit_step = 100
eval_step = 200


def load_data(path):
    return pd.read_pickle(path + "/dataframe.pkl")


def create_cnn(n_classes):

    model = Sequential()
    model.add(
        Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
            input_shape=(129, 2946, 1),
        )
    )
    #model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    #model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    #model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model


def create_split(path):
    df = load_data(path)
    df["genre"] = pd.factorize(df["genre"])[0]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(df["spectrogram"], df["genre"]):
        X_train, X_test = df["spectrogram"][train_index], df["spectrogram"][test_index]
        y_train, y_test = df["genre"][train_index], df["genre"][test_index]

    return X_train, X_test, y_train, y_test


def train_model(model, X, y):
    # Cast it to a numpy array
    X = np.array(X.tolist())
    X_train = X[:-100]
    y_train = y[:-100]
    X_val = X[-100:]
    y_val = y[-100:]
    

    # Compile model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    X_test = np.array(X_test.tolist())

    _, test_acc = model.evaluate(
        X_test,
        y_test,
        verbose=2,
        batch_size=batch_size,
    )

    print(test_acc)


def plot_accuracy(history):

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.0, 1])
    plt.legend(loc="lower right")

    plt.show()

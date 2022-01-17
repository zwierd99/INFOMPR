# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import keras

# TensorFlow Dependencies
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout

import math

# Allow GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 10
fit_step = 100
eval_step = 200


def load_data(path, pickle):
    return pd.read_pickle(path + "/" + pickle)


def create_cnn(n_classes, input):
    input_shape = (input[0].shape[0],input[0].shape[1],1)    
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation = "relu"))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (2, 2), activation = "relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (1, 1), activation = "relu"))
    model.add(MaxPool2D((1, 1), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))
        
    print(model.summary())

    return model


def create_split(path, pickle):
    df = load_data(path, pickle)
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
        optimizer=optimizers.Adam(lr=1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=15,
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


def plot_accuracy(hist):
    
    fig, axs = plt.subplots(2)
    
    # accuracy subplot
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")    
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    # Error subplot
    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="test error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
    plt.show()

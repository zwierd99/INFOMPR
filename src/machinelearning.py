# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# TensorFlow Dependencies
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

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
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model


def create_split(path):
    df = load_data(path)
    df["genre"] = pd.factorize(df["genre"])[0]
    print(df)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    for train_index, test_index in sss.split(df["spectrogram"], df["genre"]):
        X_train, X_test = df["spectrogram"][train_index], df["spectrogram"][test_index]
        y_train, y_test = df["genre"][train_index], df["genre"][test_index]

    return X_train, X_test, y_train, y_test


def train_model(model, X, y):
    # Cast it to a numpy array
    X = np.array(X.tolist())

    # Compile model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Fit data
    for i in range(0, len(X), fit_step):
        history = model.fit(
            x=X[i : i + fit_step],
            y=y[i : i + fit_step],
            epochs=1,
            batch_size=batch_size,
        )

    return model, history


def evaluate_model(model, X_test, y_test):
    X_test = np.array(X_test.tolist())

    for i in range(0, len(X_test), eval_step):
        test_loss, test_acc = model.evaluate(
            x=X_test[i : i + fit_step],
            y=y_test[i : i + fit_step],
            verbose=2,
            batch_size=batch_size,
        )

    print(test_acc)


def plot_accuracy(history):

    plt.plot(history.history["accuracy"], label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")

    plt.show()

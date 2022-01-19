# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# TensorFlow Dependencies
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Activation

# Allow GPU memory growth
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 64
learning_rate = 0.0005
epochs = 15
test_size = 0.1


def load_data(path, pickle):
    return pd.read_pickle(path + "/" + pickle)


def create_cnn(n_classes, input):
    input_shape = (input[0].shape[0],input[0].shape[1],1)    
    
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D())    

    model.add(Conv2D(16, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D())   

    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D())   

    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D())


    model.add(Flatten())
    model.add(Dropout(0.8))
    model.add(Dense(n_classes, activation='softmax'))
        
    print(model.summary())

    return model


def create_split(path, pickle):
    df = load_data(path, pickle)
    df["genre"] = pd.factorize(df["genre"])[0]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

    for train_index, test_index in sss.split(df["spectrogram"], df["genre"]):
        X_train, X_test = df["spectrogram"][train_index], df["spectrogram"][test_index]
        y_train, y_test = df["genre"][train_index], df["genre"][test_index]

    return X_train, X_test, y_train, y_test


def train_model(model, X, y):
    # Cast it to a numpy array            
    X = np.array(X.tolist())
    print((len(X)/(1-test_size)))
    validation_size = int((len(X)/(1-test_size))*test_size)
    X_train = X[:-validation_size]
    y_train = y[:-validation_size]
    X_val = X[(-validation_size):]
    y_val = y[(-validation_size):]


    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
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

    print(f"Test Accuracy: {test_acc}")
    # print(f"Test RMSE: {test_f1}")

def plot_accuracy(hist):
    
    fig, axs = plt.subplots(2)
        
    # accuracy subplot
    axs[0].plot(hist.history["accuracy"], label="Train Accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="Test Accuracy")    
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Eval")
    
    # Error subplot
    axs[1].plot(hist.history["loss"], label="Train Error")
    axs[1].plot(hist.history["val_loss"], label="Test Error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Eval")
    
    # F1 subplot
    # axs[2].plot(hist.history["root_mean_squared_error"], label="Train RMSE")
    # axs[2].plot(hist.history["val_root_mean_squared_error"], label="Test RMSE")
    # axs[2].set_ylabel("RMSE")
    # axs[2].legend(loc="lower right")
    # axs[2].set_title("RMSE Eval")
    
    plt.show()

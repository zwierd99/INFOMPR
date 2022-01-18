# 3rd party Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

# TensorFlow Dependencies
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Activation

# Allow GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size = 10
fit_step = 100
eval_step = 200
learning_rate = 0.0005
epochs = 30


def load_data(path, pickle):
    return pd.read_pickle(path + "/" + pickle)


def create_cnn(n_classes, input):
    input_shape = (input[0].shape[0],input[0].shape[1],1)    
    
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D(2, 2))    

    model.add(Conv2D(16, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=3))
    #model.add(Activation("relu"))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D(2, 2))   

    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(2, 2))   

    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(2, 2))   
        
    model.add(Flatten())
    
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
        
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

def f1_score(y_true, y_pred):
    # Count positive samples.
    true_positives  = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives  = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives  = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # How many selected items are relevant?
    precision = true_positives  / (possible_positives + K.epsilon())

    # How many relevant items are selected?
    recall = true_positives  / (predicted_positives + K.epsilon())

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score

def train_model(model, X, y):
    # Cast it to a numpy array            
    X = np.array(X.tolist())
    X_train = X[:-100]
    y_train = y[:-100]
    X_val = X[-100:]
    y_val = y[-100:]
    

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy", f1_score]
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

    _, test_acc, test_f1 = model.evaluate(
        X_test,
        y_test,
        verbose=2,
        batch_size=batch_size,
    )

    print(f"Test Accuracy: {test_acc}")
    print(f"F1 Accuracy: {test_f1}")

def plot_accuracy(hist):
    
    fig, axs = plt.subplots(3)
        
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
    axs[2].plot(hist.history["f1_score"], label="Train F1 score")
    axs[2].plot(hist.history["val_f1_score"], label="Test F1 score")    
    axs[2].set_ylabel("F1 Score")
    axs[2].legend(loc="lower right")
    axs[2].set_title("F1 Score Eval")
    
    plt.show()

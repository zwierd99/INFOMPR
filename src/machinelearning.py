import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit

PATH = 'data/dataframe.pkl'

def load_data(path):
    return pd.read_pickle(path)

def create_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(129, (3,3), activation='relu', input_shape=(129, 2946 ,1)))
    #model.add(layers.MaxPooling2D((2, 2)))

    # Add Dense layers on top
    model.add(layers.Flatten())
    model.add(layers.Dense(10))

    return model

def create_split():
    df = load_data(PATH)
    df['spectrogram'] = [x[:,:2946] for x in df['spectrogram']]
    df['genre'] = pd.factorize(df['genre'])[0]
    print(df)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    
    for train_index, test_index in sss.split(df['spectrogram'], df['genre']):
        X_train, X_test = df['spectrogram'][train_index], df['spectrogram'][test_index]
        y_train, y_test = df['genre'][train_index], df['genre'][test_index]

    return X_train, X_test, y_train, y_test

def train_model(model, X, y):
    X = np.array([np.asarray(x).astype('float32') for x in X])
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit data
    history = model.fit(X, y, epochs=1, batch_size=1)

    return model, history

def evaluate_model(model, history, X_test, y_test):
    X_test = np.array([np.asarray(x).astype('float32') for x in X_test])

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print(test_acc)
    
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

from sklearn.model_selection import StratifiedShuffleSplit

PATH = 'data/dataframe.pkl'

def load_data(path):
    return pd.read_pickle(path)

def create_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(129, (3,3), activation='relu', input_shape=(129, 2954 ,1)))
    #model.add(layers.MaxPooling2D((2, 2)))

    # Add Dense layers on top

    return model

def train_model(model):

    # Create split
    df = load_data(PATH)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    
    for train_index, test_index in sss.split(df['spectrogram'], df['genre']):
        X_train, X_test = df['spectrogram'][train_index], df['spectrogram'][test_index]
        y_train, y_test = df['genre'][train_index], df['genre'][test_index]

    print(X_train[:10], y_train[:10])    
    """
    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit data
    history = model.fit(X_train, y_train, epochs=10, 
                        validation_data=(X_test, y_test))

    #return model
    """

def evaluate_model(model, history, X_test, y_test):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    

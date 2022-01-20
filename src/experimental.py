# Default libraries
import os
import random
import shutil

# 3rd party libraries
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Tensorflow libraries
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Rescaling

# Allow GPU memory growth
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def make_pics(path):
    """
    Uses the audio files to generate mel spectograms pictures
    """
    matplotlib.use('Agg')
        
    print(f"Starting mel spectrogram generation in {path}...")
    for subdir, _, files in os.walk(path):
        # Ignore files that are not folders
        if subdir[len(path) + 1:] != '':
            genre = subdir[len(path) + 1:]
        else:
            continue
        for file in files:
            # Only process .wav files
            if file.endswith(".wav"):
                # Check if image already exists
                if not os.path.exists(f'{path+"_img"}/{genre}/{file.split(".")[0]}.png'):
                    print(file)
                    
                    # Get the audio file
                    y,sr = librosa.load(f"{path}/{genre}/{file}",duration=3)
                    
                    # Convert it to mel spectrogram
                    mels = librosa.feature.melspectrogram(y=y,sr=sr)
                    fig = plt.Figure()
                    canvas = FigureCanvas(fig)
                    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
                    
                    #librosa.display.specshow(librosa.power_to_db(mels,ref=np.max))
                    
                    # Remove padding
                    #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    #            hspace = 0, wspace = 0)
                    #plt.margins(0,0)
                    
                    # Make the folder if it doesn't exist
                    if not os.path.exists(f'{path+"_img"}/{genre}'):
                        os.makedirs(f'{path+"_img"}/{genre}')
                    plt.savefig(f'{path+"_img"}/{genre}/{file.split(".")[0]}.png')
                    plt.cla()

def train_test_split(path):
    
    for subdir, _, files in os.walk(path):
        genre = subdir[len(path) + 1:]
        
        random.shuffle(files)
        
        # Take 100 files of each genre
        test_files = files[0:100]
        
        for f in test_files:
            
            # Make the folder if it doesn't exist
            if not os.path.exists(f'{path+"_test"}/{genre}'):
                os.makedirs(f'{path+"_test"}/{genre}')
            shutil.move(f"{path}/{genre}/{f}",f'{path+"_test"}/{genre}/{f}')
            
            print("Moved file:", f)

# https://www.tensorflow.org/tutorials/load_data/images
def generators(path):
    path += '_img'
    
    # For rescaling the data to values betwen 0-1
    normalization_layer = Rescaling(1./255)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.1,
    subset="training",
    color_mode='rgba',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(lambda x, y: (normalization_layer(x), y))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.1,
    subset="validation",
    color_mode='rgba',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(lambda x, y: (normalization_layer(x), y))
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
    path+'_test',
    validation_split=0,
    color_mode='rgba',
    image_size=(img_height, img_width),
    batch_size=batch_size).map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds, test_ds

def model():
    input_shape = (img_height, img_width, 4)
    n_classes = 10    
    
    model = Sequential()
    
    # For rescaling the data   
    # could try leaky relu
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
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
        
    print(model.summary())

    return model


def train_model(model, train, validation):
    
    # Compile model
    model.compile(
        # Maybe try tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer=optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        train,
        epochs=epochs,
        validation_data=validation
    )
    
    return model, history


def evaluate_model(model, test):
    
    _, test_acc = model.evaluate(
        test,
    )

    print(f"Test Accuracy: {test_acc}")
    

def plot_accuracy(hist):
    
    _, axs = plt.subplots(2)
        
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

    plt.show()
  
    
def setup_files(PATH):
    make_pics(PATH)
    train_test_split(PATH+'_img')
        

if __name__ == "__main__":
    
    # Global variables
    PATH = "F:/data_3sec"
    batch_size=128
    learning_rate=0.005
    epochs=70
    img_height=432
    img_width=288
    
    # Uncomment if you do not have the images + test files
    #setup_files(PATH)
    
    train, val, test = generators(PATH)
    cnn = model()
    trained_model, history = train_model(cnn, train, val)
    evaluate_model(trained_model, test)
    plot_accuracy(history)
    
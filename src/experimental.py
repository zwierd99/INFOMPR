import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import shutil
import matplotlib
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Activation

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
        if subdir[len(path) + 1:] != '':
            genre = subdir[len(path) + 1:]
        else:
            continue
        for file in files:
            if file.endswith(".wav"):
                # Check if image already exists
                if not os.path.exists(f'{path+"_img"}/{genre}/{file.split(".")[0]}.png'):
                    print(file)
                    y,sr = librosa.load(f"{path}/{genre}/{file}",duration=3)
                    mels = librosa.feature.melspectrogram(y=y,sr=sr)
                    fig = plt.Figure()
                    FigureCanvas(fig)
                    plt.imshow(librosa.power_to_db(mels,ref=np.max))
                    plt.savefig(f'{path+"_img"}/{genre}/{file.split(".")[0]}.png')
                    plt.cla()


def train_test_split(path):
    
    for subdir, _, files in os.walk(path):
        genre = subdir[len(path) + 1:]
        
        random.shuffle(files)
        test_files = files[0:100]
        
        for f in test_files:
            print("Moved file:", f)
            shutil.move(f"{path}/{genre}/{f}",f'{path+"_test"}/{genre}/{f}')

# https://www.tensorflow.org/tutorials/load_data/images
def generators(path):
    img_height=640#288
    img_width=480#432
    batch_size=128
    
    # For rescaling the data
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=0.2,
    subset="training",
    color_mode='rgba',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(lambda x, y: (normalization_layer(x), y))
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
    path+"_test",
    validation_split=0.2,
    subset="validation",
    color_mode='rgba',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size).map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds

def model():
    input_shape = (640, 480, 4)
    n_classes = 10    
    
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
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='softmax'))
        
    print(model.summary())

    return model

def train_model(model, train, validation):
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.005), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        train,
        epochs=70,
        validation_data=validation
    )
    
    return model, history

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

    plt.show()
    
if __name__ == "__main__":
    PATH = "F:/data_3sec_img"
    #make_pics(PATH)
    #train_test_split(PATH)
    train, test = generators(PATH)
    cnn = model()
    trained_model, history = train_model(cnn, train, test)
    plot_accuracy(history)
    
    
# streamlit run src/demo.py
import os

import streamlit as st
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    BatchNormalization,
)

def create_cnn(n_classes):
    input_shape = (128, 130, 1)

    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation="relu", input_shape=input_shape, padding="same"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D(padding="same"))

    model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D(padding="same"))

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization(axis=3))
    model.add(MaxPool2D(padding="same"))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(padding="same"))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(padding="same"))

    model.add(Flatten())
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    return model

def get_model():  
    """ Loads the model """
    model = create_cnn(10)  
    model.load_weights('model_weights/best_weights.h5')
    return model

def convert_mp3_to_wav(music_file):  
    """ Converts .mp3 files to .wav format """
    sound = AudioSegment.from_mp3(music_file)      
    sound.export("music_file.wav",format="wav")
    
def extract(file,t1,t2):  
    """ Extracts the audio file from the given time interval """
    convert_mp3_to_wav(file)  
    
    count = 0
    for i in range(t1, t2, 3):
        sound = AudioSegment.from_wav("music_file.wav")
        sound = sound[i*1000:i*1000+3000]
        sound.export(f"{count}_extracted.wav", format="wav")
        count += 1
    
def get_features():  
    """ Extracts features from the audio file """
       
    mels = []
    
    for i in range(end//3):
        y,sr = librosa.load(f"{i}_extracted.wav",duration=3)  
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
        mel_spec_arr = np.asarray(mel_spec).astype("float32")
        mfcc = librosa.feature.mfcc(S=mel_spec)
        mfcc = np.asarray(mfcc).astype("float32")
        mels.append(mel_spec_arr)
        
    return mels, mfcc
    
model = get_model()

def predict_part(data):
    prediction = model.predict(np.expand_dims(data, 0))  
    prediction = prediction.reshape((10,))     
    class_label = np.argmax(prediction) 
    
    return class_label

def predict(file_name, data):    
    song = file_name.split('.')[0]
    
    predictions = []
    
    for d in data:
        predictions.append(predict_part(d))

    # Get most common prediction
    probabilities = []
    for i in range(10):
        occurence = predictions.count(i)
        probabilities.append(occurence/len(predictions))

    print(probabilities)
    prediction = np.argmax(probabilities)
            
    st.write(f"## The Genre of {song} is "+class_labels[prediction])
    
    # Show probability distribution
    color_data = [1,2,3,4,5,6,7,8,9,10]
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=10)

    fig,ax= plt.subplots(figsize=(6,4.5))
    ax.bar(x=class_labels,height=probabilities,
    color=my_cmap(my_norm(color_data)))
    plt.xticks(rotation=45)
    ax.set_title(f"Probability Distribution Of {song} Over Different Genres")

    st.pyplot(fig)
    
    # Show the mel spectrogram
    fig2, ax2 = plt.subplots(figsize=(10,4))
    y, sr = librosa.load("music_file.wav", duration=end)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    librosa.display.specshow(mel, y_axis='mel', sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram of {song}')
    plt.tight_layout()
    
    st.pyplot(fig2)


class_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

file = st.sidebar.file_uploader("Please Upload An Mp3 Audio File Here",type=["mp3"])

start = 0
end = 120

if file is None:  
    st.text("Please upload an mp3 file")
else:  
    
    extract(file,start,start+end)   
    mel_spec, mfcc = get_features()   
    predict(file.name, mel_spec)   
    
    os.remove("music_file.wav")
    for i in range(10):
        os.remove(f"{i}_extracted.wav")
    
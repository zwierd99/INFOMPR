import cv2
import os
import pandas as pd
import gc
import librosa
import librosa.display  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler



def loop_files():
    df = pd.DataFrame(columns=["name", "genre", "spectrogram"])
    for subdir, _, files in os.walk("pngs"):
        for file in files:
            genre = subdir[len("pngs") + 1:]
            img = cv2.imread(f"pngs/{genre}/{file}", 0)
            cropped_img = img[35:35 + 217, 55:335 + 55]
            # New img size is 217x335

            df = df.append(
                {
                    "name": file,
                    "genre": genre,
                    "spectrogram": cropped_img,
                },
                ignore_index=True,
            )
            print(f"/{genre}/{file}")
            gc.collect()

    df.to_pickle("mel_spectrograms.pkl")
    
    
def make_pickle(path):
    """
    Uses the audio files to generate mel spectograms
    """
    
    df = pd.DataFrame(columns=["genre", "spectrogram"])
    
    print(f"Starting mel spectrogram generation in {path}...")
    for subdir, _, files in os.walk(path):
        if subdir[len(path) + 1:] != '':
            genre = subdir[len(path) + 1:]
        else:
            continue
        for file in files:
            if file.endswith(".wav"):
                print(file)
                y,sr = librosa.load(f"{path}/{genre}/{file}",duration=3)
                # Ranges between -80 and 0
                mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr),ref=np.max)
                mel_spec = np.asarray(mel_spec).astype("float32")
                if mel_spec.shape == (128,130):
                    df = df.append({"genre" : genre,
                                    "spectrogram": mel_spec},
                                    ignore_index=True)

    

    df.to_pickle(f"{path}/mel_spectrograms.pkl")
    print("Done")
    
    # Show first mel spectrogram
    plt.figure(figsize=(10,4))  
    librosa.display.specshow(df.iloc[0]['spectrogram'], y_axis='mel', sr=sr, x_axis='time')
    plt.colorbar(format='%+2.0f dB')  
    plt.title('Mel-Spectrogram')  
    plt.tight_layout()  
    plt.show()  

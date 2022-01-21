# Source: https://www.kaggle.com/dapy15/music-genre-classification

import math
import os

import numpy as np
import librosa
import pandas as pd

def make_pickle(path):
    """
    Uses the audio files to generate mel spectograms
    """
    
    df = pd.DataFrame(columns=["genre", "spectrogram"])
    
    print(f"Starting mfcc generation in {path}...")
    for subdir, _, files in os.walk(path):
        if subdir[len(path) + 1:] != '':
            genre = subdir[len(path) + 1:]
        else:
            continue
        for file in files:
            if file.endswith(".wav"):
                print(file)
                                
                y,sr = librosa.load(f"{path}/{genre}/{file}",duration=3)
                mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr),ref=np.max)
                
                mfcc = librosa.feature.mfcc(S=mel_spec)
                mfcc = np.asarray(mfcc).astype("float32")
                
                if mfcc.shape == (20,130):
                    df = df.append({"genre" : genre,
                                    "spectrogram": mfcc},
                                    ignore_index=True)


    df.to_pickle(f"{path}/mfcc.pkl")
    print("Done")
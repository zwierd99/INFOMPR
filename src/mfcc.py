# Source: https://www.kaggle.com/dapy15/music-genre-classification

import math
import os

import numpy as np
import librosa
import pandas as pd

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
n_mfcc=13
n_fft=2048
hop_length=512 
num_segments=5

def make_pickle(path):
    """
    Uses the audio files to generate mel spectograms
    """
    
    df = pd.DataFrame(columns=["genre", "mfcc"])
    
    print(f"Starting mfcc generation in {path}...")
    for subdir, _, files in os.walk(path):
        if subdir[len(path) + 1:] != '':
            genre = subdir[len(path) + 1:]
        else:
            continue
        for file in files:
            if file.endswith(".wav"):
                print(file)
                
                samples_ps = int(SAMPLES_PER_TRACK/num_segments) # ps = per segment
                expected_vects_ps = math.ceil(samples_ps/hop_length)
                
                y,sr = librosa.load(f"{path}/{genre}/{file}",duration=3)
                mfcc = np.asarray(librosa.feature.mfcc(y,
                                    sr = sr,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length)).astype("float32")

                mfcc = mfcc.T
                
                # store mfcc if it has expected length 
                if len(mfcc)==expected_vects_ps:
                    df = df.append({"genre" : genre,
                                    "mfcc": mfcc},
                                    ignore_index=True)


    df.to_pickle(f"{path}/mfcc.pkl")
    print("Done")
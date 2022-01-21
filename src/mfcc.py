# Source: https://www.kaggle.com/dapy15/music-genre-classification

import os

import numpy as np
import librosa
import pandas as pd


def make_combined_pickle(path):
    """
    Uses the audio files to generate mel spectograms
    """

    df = pd.DataFrame(columns=["genre", "spectrogram", "mfcc"])

    print(f"Starting mfcc generation in {path}...")
    for subdir, _, files in os.walk(path):
        if subdir[len(path) + 1 :] != "":
            genre = subdir[len(path) + 1 :]
        else:
            continue
        for file in files:
            if file.endswith(".wav"):
                print(file)

                y, sr = librosa.load(f"{path}/{genre}/{file}", duration=3)
                mel_spec = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max
                )

                mel_spec_arr = np.asarray(mel_spec).astype("float32")

                mfcc = librosa.feature.mfcc(S=mel_spec)
                mfcc = np.asarray(mfcc).astype("float32")

                if mfcc.shape == (20, 130) and mel_spec_arr.shape == (128, 130):
                    df = df.append(
                        {"genre": genre, "spectrogram": mel_spec_arr, "mfcc": mfcc},
                        ignore_index=True,
                    )

    df.to_pickle(f"{path}/mfcc_and_spectrogram.pkl")
    print("Done")

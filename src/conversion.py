import gc
import time
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_wav_file(genre, sample_name):
    imagepath = f"spectrograms/{genre}/{sample_name}.png"
    try:
        os.mkdir(f"spectrograms/{genre}/")
    except:
        pass
    plt.savefig(imagepath, bbox_inches="tight")


def spectrogram_array(path, genre, wav_file):
    print(f"{path}/{genre}/{wav_file}")
    sample_rate, samples = wavfile.read(f"{path}/{genre}/{wav_file}")

    # Cut it off at the smallest possible length and cast to float32
    return np.asarray(signal.spectrogram(samples, sample_rate)[2][:, :2946]).astype(
        "float32"
    )


def graph_wav_file(path, genre, wav_file, sample_name):
    plt.clf()
    print(f"{path}/{genre}/{wav_file}")
    sample_rate, samples = wavfile.read(f"{path}/{genre}/{wav_file}")
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plot = plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.axis("off")
    save_wav_file(genre, sample_name)


def convert_GZTAN(path):
    for subdir, _, files in os.walk(path):
        for file in files:
            start = time.time()
            genre = subdir[len(path) + 1 :]
            filename = file[:-4]
            print(f"spectrograms/{genre}/{filename}.png")
            if not os.path.exists(f"spectrograms/{genre}/{filename}.png"):
                graph_wav_file(path, genre, file, filename)
                gc.collect()
                end = time.time()
                print(f"{end-start} seconds passed.")


def spectrogram_pickle(path):
    df = pd.DataFrame(columns=["name", "genre", "spectrogram"])

    print("Generating spectrogram pickle...")
    for subdir, _, files in os.walk(path):
        for file in files:
            genre = subdir[len(path) + 1 :]
            df = df.append(
                {
                    "name": file,
                    "genre": genre,
                    "spectrogram": spectrogram_array(path, genre, file),
                },
                ignore_index=True,
            )
            gc.collect()

    df.to_pickle(f"{path}/dataframe.pkl")
    print("Done")

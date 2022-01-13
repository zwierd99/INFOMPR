import gc
import time
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_wav_file(genre, sample_name):
    imagepath = f'spectrograms/{genre}/{sample_name}.png'
    try:
        os.mkdir(f'spectrograms/{genre}/')
    except:
        pass
    plt.savefig(imagepath, bbox_inches='tight')


def spectrogram_array(genre, wav_file):
    print(f'data/{genre}/{wav_file}')
    sample_rate, samples = wavfile.read(f'data/{genre}/{wav_file}')

    return signal.spectrogram(samples, sample_rate)[2]


def graph_wav_file(genre, wav_file, sample_name):
    plt.clf()
    print(f'data/{genre}/{wav_file}')
    sample_rate, samples = wavfile.read(f'data/{genre}/{wav_file}')
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plot = plt.pcolormesh(times, frequencies, np.log(spectrogram))
    plt.axis('off')
    save_wav_file(genre, sample_name)


def convert_GZTAN(path):
    df = pd.DataFrame(columns=["name", "genre", "spectrogram"])

    for subdir, _, files in os.walk(path):
        for file in files:
            start = time.time()
            genre = subdir[5:]
            filename = file[:-4]
            print(f'spectrograms/{genre}/{filename}.png')
            if not os.path.exists(f'spectrograms/{genre}/{filename}.png'):
                df = df.append({'name': file, 'genre': genre, 'spectrogram': spectrogram_array(genre, file)}, ignore_index=True)
                graph_wav_file(genre, file, filename)
                gc.collect()
                end = time.time()
                print(f'{end - start} seconds passed.')

    df.to_pickle('data/dataframe.pkl')


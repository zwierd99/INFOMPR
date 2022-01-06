import gc
import time
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import matplotlib.pyplot as plt


def create_plot(genre, file):
    plt.clf()
    sample_rate, samples = wavfile.read(f'data/{genre}/{file}')
    frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)

    plot = plt.pcolormesh(times, frequencies, np.log(spectogram))
    plt.axis('off')
    return(plot)
    # plt.imshow(spectrogram)


def convert_GZTAN(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            start = time.time()
            genre = subdir[5:]
            filename = file[:-4]
            print(f'spectograms/{genre}/{filename}.png')
            if not os.path.exists(f'spectograms/{genre}/{filename}.png'):
                create_plot(genre, file)



                imagepath = f'spectograms/{genre}/{filename}.png'
                try:
                    os.mkdir(f'spectograms/{genre}/')
                except:
                    pass
                plt.savefig(imagepath, bbox_inches='tight')
                gc.collect()
                end = time.time()
                print(end-start)

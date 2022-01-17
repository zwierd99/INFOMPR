#!/usr/bin/env python3
from numpy import unique

# Local dependencies
import datastatistics as ds
import conversion as cv
import machinelearning as ml
import mel_spectrograms as ms

PATH = "data"
PICKLE = 'mel_spectrograms.pkl'
#PICKLE = 'dataframe.pkl'


def cnn():
    X_train, X_test, y_train, y_test = ml.create_split(PATH, PICKLE)
    model, his = ml.train_model(ml.create_cnn(len(unique(y_train)), X_train), X_train, y_train)

    ml.evaluate_model(model, X_test, y_test)
    ml.plot_accuracy(his)


if __name__ == "__main__":
    # cv.spectrogram_pickle(PATH)
    # ms.loop_files()

    cnn()

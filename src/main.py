#!/usr/bin/env python3
# Local Libraries
import os

# 3rd Party Dependencies
import numpy as np

# Local dependencies
import conversion as cv
import machinelearning as ml
import mfcc

PATH = "data/data_10sec"
SHORT_PATH = "data/data_3sec"
PICKLE = "mfcc_and_spectrogram.pkl"
CHECKPOINT = "model_weights/best_weights.h5"

if __name__ == "__main__":
    if not os.path.exists(f"{SHORT_PATH}/{PICKLE}"):
        cv.generate_3sec(PATH)
        mfcc.make_combined_pickle(SHORT_PATH)

    X_train, X_test, y_train, y_test = ml.create_split(SHORT_PATH, PICKLE)
    model = ml.create_cnn(len(np.unique(y_train)), X_train)

    if not os.path.exists(CHECKPOINT):
        model, his = ml.train_model(model, X_train, y_train)
        ml.plot_accuracy(his)

    ml.evaluate_model(model, X_test, y_test, CHECKPOINT)

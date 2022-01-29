#!/usr/bin/env python3
# Local Libraries
import os

# 3rd Party Dependencies
import numpy as np

# Local dependencies
import src.conversion as cv
import src.machinelearning as ml
import src.mfcc as mfcc
import src.evaluation as eval
from tensorflow.keras import Sequential, optimizers

PATH = "data/data_10sec"
SHORT_PATH = "data/data_3sec"
PICKLE = "mfcc_and_spectrogram.pkl"

CHECKPOINT = "model_weights/best_weights.h5"
INPUT_TYPE = "spectrogram" #spectrogram or mfcc


def mcnemar_setup():
    X_train, X_test, y_train, y_test = ml.create_split(SHORT_PATH, PICKLE, "spectrogram")
    model_mel = ml.create_cnn(len(np.unique(y_train)), X_train)

    model_mel.compile(
        optimizer=optimizers.Adam(learning_rate=ml.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model_mel.load_weights("model_weights/best_weights.h5")
    X_test = np.array(X_test.tolist())
    mel_prob = model_mel.predict(X_test)
    mel_pred = mel_prob.argmax(axis=-1)


    X_train, X_test, y_train, y_test = ml.create_split(SHORT_PATH, PICKLE, "mfcc")
    model_mfcc = ml.create_cnn(len(np.unique(y_train)), X_train)
    model_mfcc.compile(
        optimizer=optimizers.Adam(learning_rate=ml.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model_mfcc.load_weights("model_weights/best_weights_mfcc.h5")
    X_test = np.array(X_test.tolist())
    mfcc_prob = model_mfcc.predict(X_test)
    mfcc_pred = mfcc_prob.argmax(axis=-1)

    eval.mcnemar_test(y_test, mel_pred, mfcc_pred)

if __name__ == "__main__":
    if not os.path.exists(f"{SHORT_PATH}"):
        try:
            os.mkdir(f"{SHORT_PATH}")
        except:
            pass
    if not os.path.exists(f"{SHORT_PATH}/{PICKLE}"):
        cv.generate_3sec(PATH)
        mfcc.make_combined_pickle(SHORT_PATH)

    X_train, X_test, y_train, y_test = ml.create_split(SHORT_PATH, PICKLE, INPUT_TYPE)
    model = ml.create_cnn(len(np.unique(y_train)), X_train)

    if not os.path.exists(CHECKPOINT):
        model, his = ml.train_model(model, X_train, y_train)
        ml.plot_accuracy(his)

    eval.evaluate_model(model, X_test, y_test, CHECKPOINT)
    mcnemar_setup()


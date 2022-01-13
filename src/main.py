#!/usr/bin/env python3

import datastatistics as ds
import conversion as cv
import machinelearning as ml

PATH = 'data'

def cnn():
    X_train, X_test, y_train, y_test = ml.create_split(PATH)
    model, his = ml.train_model(ml.create_cnn(), X_train, y_train)
    ml.evaluate_model(model, his, X_test, y_test)

if __name__ == '__main__':
    # files = ['data/features_3_sec.csv', 'data/features_30_sec.csv']
    # labels, digits = ds.extract_data(files)
    #cv.convert_GZTAN('data')
    cv.spectogram_pickle(PATH)
    
    #cnn()
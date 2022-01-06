#!/usr/bin/env python3

import datastatistics as ds

if __name__ == '__main__':
    files = ['data/features_3_sec.csv', 'data/features_30_sec.csv']
    labels, digits = ds.extract_data(files)
    
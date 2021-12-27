import pandas as pd
import numpy as np
def extract_data(files):
    print(files)
    gztan_data = [pd.read_csv(filename) for filename in files]
    print(gztan_data[0]['label'])
    


def main():
    files = ['features_3_sec.csv', 'features_30_sec.csv']
    labels, digits = extract_data(files)
    


if __name__ == '__main__':
    main()
        
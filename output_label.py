import pandas as pd
import numpy as np


if __name__ == '__main__':
    '''
    This script is justed used for outputing the labels csv file.
    output:
        image name -> index id
    '''
    data_train = pd.read_csv('./input/train.csv')
    names_train = data_train['Image'].values
    labels_train = data_train['Id'].values

    unique_labels_value = np.unique(labels_train)
    label_csv_df = pd.DataFrame({'Image': unique_labels_value})
    label_csv_df['Id'] = label_csv_df.index

    # print(label_csv_df.head())
    label_csv_df.to_csv('./input/label.csv', index=False)

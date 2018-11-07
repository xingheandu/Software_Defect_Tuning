from __future__ import division
import pandas as pd
import numpy as np

DATASET_PATH = r"C:\Users\terry\Documents\Software_Defect_Tuning\testDataset\ant-1.3.csv"


def read_data(path):
    """
    Read the data into pandas dataframe
    :param path:
    :return:
    """
    data = pd.read_csv(path, skipinitialspace=True)
    data = data.iloc[:, 3:]
    return data


def dataset_statistics(dataset):
    """
    Basic statistics of the dataset
    :param dataset: Pandas dataframe
    :return: None, print the basic statistics of the dataset
    """
    # print(dataset)
    count_row = dataset.shape[0]
    print("Total # of rows:", count_row)
    defect = dataset.iloc[:, -1]
    # print(defect)
    num_defect = np.count_nonzero(defect)
    print("Total # of defects:", num_defect)
    defect_percentage = num_defect / count_row
    print("% of defects: ", defect_percentage)


def main():
    dataset = read_data(DATASET_PATH)
    dataset_statistics(dataset)


if __name__ == "__main__":
    main()

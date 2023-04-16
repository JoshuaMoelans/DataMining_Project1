import pandas as pd

import DT
import KNN
import RFC
from visualizer import visualize_columns_2


def get_csv_data(file_path):
    """
    Read the data from a csv file and return a list of data points
    :param file_path: path to the csv file
    :return: list of data points
    """
    data = pd.read_csv(file_path)
    return data


if __name__ == '__main__':
    data = get_csv_data("./data/existing-customers.csv")
    # visualize_columns_2(data)
    # print("estimated ROI: ",DT.predict_ROI_DT(data))
    # print("estimated ROI: ", RFC.predict_ROI_RFC(data))
    print("estimated ROI: ", KNN.predict_ROI_KNN(data))
    # DT.test_decision_tree_approach(data)
    # RFC.test_RFC_approach(data)
    # KNN.test_knn_approach(data)
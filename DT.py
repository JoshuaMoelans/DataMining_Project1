import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from labeller import label_potential_customers


def read_existing_customers_DT(data):
    """
    Read the existing customers data and return a classifier
    Based on https://www.datacamp.com/tutorial/decision-tree-classification-python
    :param data: the input dataset to train/validate the classifier
    :return: classifier, score, estimated profit, precision%, feature_cols
    """
    # feature_cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
    #                 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    feature_cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
    # feature_cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation',
    #                 'relationship', 'capital-gain', 'capital-loss', 'hours-per-week']
    # feature_cols = ['age', 'workclass', 'education-num', 'marital-status', 'occupation',
    #                 'relationship', 'capital-gain', 'capital-loss', 'hours-per-week']
    # feature_cols = ['age', 'education-num', 'marital-status', 'occupation',
    #                 'relationship', 'capital-gain', 'capital-loss', 'hours-per-week']
    # feature_cols = ['capital-gain', 'education-num', 'marital-status']
    # convert non-numerical features to numerical
    X = pd.get_dummies(data[feature_cols], drop_first=True)
    Y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)  # 70% training and 30% test
    # Create Decision Tree classifier object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifier
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    profit = 0.0
    y = 0  # total positive predictions
    for i in range(len(y_pred)):
        if y_pred[i] == '>50K':
            y += 1
            if y_pred[i] == y_test.iloc[i]:
                profit += 88.0  # avg profit per True Positive
            else:
                profit -= 25.5  # avg profit (loss) per False Positive
    # ROI calculation = (88x-25.5(1-x))*y with x = precision and y = total positive predictions
    precision = metrics.precision_score(y_test, y_pred, pos_label='>50K')
    # get the decision tree
    # actual_columns = X.columns
    # fig = plt.figure(figsize=(25, 20))
    # _ = tree.plot_tree(clf,
    #                    feature_names=actual_columns,
    #                    class_names=['<=50K', '>50K'],
    #                    filled=True)
    # fig.savefig("./results/decision_tree.svg")
    return clf, score, profit, precision, feature_cols


def test_decision_tree_approach(data):
    all_scores = []
    all_profits = []
    all_precisions = []
    for i in range(25):
        _, score, profit, precision, fc = read_existing_customers_DT(data)
        all_scores.append(score)
        all_profits.append(profit)
        all_precisions.append(precision)
    print("Average precision: ", np.average(all_precisions))
    print("Median precision: ", np.median(all_precisions))
    print("Max precision: ", np.max(all_precisions))
    print("Min precision: ", np.min(all_precisions))
    # print("Average accuracy: ", np.average(all_scores))
    # print("Median accuracy: ", np.median(all_scores))
    # print("Max accuracy: ", np.max(all_scores))
    # print("Min accuracy: ", np.min(all_scores))
    # print()
    # print("Average profit: ", np.average(all_profits))
    # print("Median profit: ", np.median(all_profits))
    # print("Max profit: ", np.max(all_profits))
    # print("Min profit: ", np.min(all_profits))


def predict_ROI_DT(data):
    clf, score, profit, precision, fc = read_existing_customers_DT(data)
    ROI, rowIDs = label_potential_customers(clf, precision, fc)
    # write rowIDs to file
    with open("./results/rowIDs.txt", "w") as f:
        for rowID in rowIDs:
            f.write(str(rowID) + ",")
    return ROI

import pandas as pd


def label_potential_customers(classifier, precision, feature_cols):
    data = pd.read_csv("./data/potential-customers.csv")
    # do something with data based on given classifier,' and return list of customers that get sent the promotion
    # ROI calculation = (88x-25.5(1-x))*y with x = precision and y = total positive predictions
    # precision gotten from classifier training (passed as parameter)
    x = precision
    y = 0
    rowIDs = []
    # limit input data to only the columns that were used in the classifier
    X_input = pd.get_dummies(data[feature_cols], drop_first=True)
    # use classifier to label potential customers
    y_pred = classifier.predict(X_input)
    # count total assignments >50K; store the rowIDs for the report
    for i in range(len(y_pred)):
        if y_pred[i] == '>50K':
            y += 1
            rowIDs.append(i)
    ROI = (88.0 * x - 25.5 * (1.0 - x)) * y
    return ROI, rowIDs

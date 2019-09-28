import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

mm_scaler = preprocessing.MinMaxScaler()

def load_train_data(normalize=False):
    train = pd.read_csv("custom_impute_train.csv", low_memory=False)

    train = train.to_numpy()

    y = train[:, 1]

    X = np.zeros((train.shape[0], 8))

    # pClass
    X[:, 0] = train[:, 2]

    # Sex
    for i in range(X.shape[0]):
        if (train[i, 4] == "male"):
            X[i, 1] = 1
        else:
            X[i, 1] = 0

    # Age
    X[:, 2] = train[:, 5]

    # SibSp
    X[:, 3] = train[:, 6]

    # Parch
    X[:, 4] = train[:, 7]

    # Fare
    X[:, 5] = train[:, 9]

    # Cabin section (assume cardinality)
    for i in range(X.shape[0]):
        element = train[i, 10]

        if type(element) == float:
            X[i, 6] = np.nan
        elif "A" in element:
            X[i, 6] = 1
        elif "B" in element:
            X[i, 6] = 2
        elif "C" in element:
            X[i, 6] = 3
        elif "D" in element:
            X[i, 6] = 4
        elif "E" in element:
            X[i, 6] = 5
        elif "F" in element:
            X[i, 6] = 6
        elif "G" in element:
            X[i, 6] = 7

    # Cabin number
    for i in range(X.shape[0]):

        if type(train[i, 10]) == float:
            X[i, 7] = np.nan
        else:
            if len(re.findall(r'\d+', train[i, 10])) == 0:
                X[i, 7] = np.nan
            else:
                X[i, 7] = int(re.findall(r'\d+', train[i, 10])[0])



    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)

    if normalize:
        X = mm_scaler.transform(X)
        mm_scaler.fit(X, y=y)

    return X, y.astype('int')

def load_test_data(normalize=False):
    train = pd.read_csv("custom_impute_test.csv", low_memory=False)

    train = train.to_numpy()

    X = np.zeros((train.shape[0], 8))

    # pClass
    X[:, 0] = train[:, 1]

    # Sex
    for i in range(X.shape[0]):
        if (train[i, 3] == "male"):
            X[i, 1] = 1
        else:
            X[i, 1] = 0

    # Age
    X[:, 2] = train[:, 4]

    # SibSp
    X[:, 3] = train[:, 5]

    # Parch
    X[:, 4] = train[:, 6]

    # Fare
    X[:, 5] = train[:, 8]

    # Cabin section (assume cardinality)
    for i in range(X.shape[0]):
        element = train[i, 9]
        if type(element) == float:
            X[i, 6] = np.nan
        elif "A" in element:
            X[i, 6] = 1
        elif "B" in element:
            X[i, 6] = 2
        elif "C" in element:
            X[i, 6] = 3
        elif "D" in element:
            X[i, 6] = 4
        elif "E" in element:
            X[i, 6] = 5
        elif "F" in element:
            X[i, 6] = 6
        elif "G" in element:
            X[i, 6] = 7

    # Cabin number
    for i in range(X.shape[0]):

        if type(train[i, 9]) == float:
            X[i, 7] = np.nan
        else:
            if len(re.findall(r'\d+', train[i, 9])) == 0:
                X[i, 7] = np.nan
            else:
                X[i, 7] = int(re.findall(r'\d+', train[i, 9])[0])

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X = imp.transform(X)
    if normalize:
        X = mm_scaler.transform(X)
    return X


def save_predictions(predictions, name="submission.csv", fmt="%i"):
    ids = pd.read_csv("test.csv", low_memory=False).to_numpy()

    ids = ids[:, 0].astype('int')
    data = np.zeros((ids.shape[0], 2))
    data[:, 0] = ids
    data[:, 1] = predictions
    np.savetxt(name, data, fmt=['%i', fmt], delimiter=",", header="PassengerId,Survived", comments='')

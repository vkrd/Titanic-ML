import numpy as np
import pandas as pd
import re

train = pd.read_csv("train.csv", low_memory=False)

train = train.to_numpy()

y = train[:, 1]

X = np.zeros((891, 11))

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
    if type(train[i, 10]) == float:
        X[i, 6] = np.nan
    elif "A" in train[i, 10]:
        X[i, 6] = 1
    elif "B" in train[i, 10]:
        X[i, 6] = 2
    elif "C" in train[i,10]:
        X[i, 6] = 3
    elif "D" in train[i, 10]:
        X[i, 6] = 4
    elif "E" in train[i, 10]:
        X[i, 6] = 5
    elif "F" in train[i,10]:
        X[i, 6] = 6
    elif "G" in train[i, 10]:
        X[i, 6] = 7

# Cabin number
for i in range(X.shape[0]):

    if type(train[i, 10]) == float:
        X[i, 7] = np.nan
    else:
        #group = train[i, 10].split(" ")
        print(len(re.findall(r'\d+', train[i, 10])))
        if len(re.findall(r'\d+', train[i, 10])) == 0 :
            X[i, 7] = np.nan
        else:
            X[i, 7] = int(re.findall(r'\d+', train[i, 10])[0])

# Embarked (one hot)
for i in range(X.shape[0]):
    if train[i, 11] == "Q":
        X[i, 8] = 1
        X[i, 9] = 0
        X[i, 10] = 0
    elif train[i, 11] == "C":
        X[i, 8] = 0
        X[i, 9] = 1
        X[i, 10] = 0
    elif train[i, 11] == "S":
        X[i, 8] = 0
        X[i, 9] = 0
        X[i, 10] = 1

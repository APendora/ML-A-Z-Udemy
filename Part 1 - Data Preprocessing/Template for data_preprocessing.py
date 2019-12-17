# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:38:31 2019

@author: squarepantsponge
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
# Uses the mean method, mean of the columns in order to fill in missing data
# Python indexes include the lower bound (Column Age) but exclude upper bound (3 is the salary column)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit (X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Categorical Data includes categories, not Num
# Encoding Categorical Data:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X [: , 0] = labelencoder_X.fit_transform(X[: , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y= labelencoder_Y.fit_transform(Y)

# Splitting dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling - Standardization vs Normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

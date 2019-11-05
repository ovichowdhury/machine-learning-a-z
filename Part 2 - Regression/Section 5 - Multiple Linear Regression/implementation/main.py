# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("50_Startups.csv")

y_index = dataset.shape[1] - 1

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, y_index].values


labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
x_cat = list(labelencoder_x.classes_)

onehot_x = OneHotEncoder(categorical_features = [3])
x = onehot_x.fit_transform(x).toarray()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

one_pred = regressor.predict(np.array([[0, 0, 1, 100000, 500, 100000]]))





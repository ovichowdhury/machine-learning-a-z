# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)


#y_pred = regressor.predict(x_test)

# plotting result 

"""
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title("Salary vs Experience (On Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

"""

from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/predict/<year>', methods=['GET'])
def predict(year):
    ar = np.array([[float(year)]])
    s = regressor.predict(ar)
    return jsonify({"salary": int(s[0])})



# Running the app
app.run(host = '0.0.0.0', port = 5000)





 

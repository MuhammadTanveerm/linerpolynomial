import pandas as pd
import numpy as mum
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data_set=pd.read_csv('Book.csv');
x= data_set.iloc[:, :1].values
y= data_set.iloc[:, 1].values
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)
regressor= LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)
x_pred= regressor.predict(x_train)
y_pred= regressor.predict(x_test)
x_pred= regressor.predict(x_train)
mtp.scatter(x_train, y_train, color="green")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Salary vs Year")

mtp.show()
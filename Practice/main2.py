import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("Practice\Housing\Housing.csv")
data = data[["price","area","bedrooms","bathrooms","stories"]]

predict = "bathrooms"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

best = 0
"""
for _ in range(100):

    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)

    if acc > best:
        best = acc 
        with open("Practice/PriceModel.pickle", "wb") as f:
            pickle.dump(linear, f)

"""

pickle_in = open("Practice/PriceModel.pickle", "rb")
linear = pickle.load(pickle_in) 
print("Coeffient: ", linear.coef_)
print('intercept: ', linear.intercept_)

var = 'bedrooms'
style.use("ggplot")
pyplot.scatter(data[var], data[predict])
pyplot.xlabel(var)
pyplot.ylabel(predict)
pyplot.show()
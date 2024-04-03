import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


#reads the info from the file and only takes from the  six variables chosen
data = pd.read_csv("Follow_along\student\student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]

#the variable we want to find
predict = "G3"

#We drop G3 from our collected data because that is what we want to find
X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

#we split our data into training and testing parts so we don't train and test the same data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size = 0.1)

"""
#this test our data 100 times and chooses the best model out of them
best = 0
for _ in range(100):

    #Here we are finding the line of best fit and printing how accurate it is
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        #creates pickle file to save our model 
        with open("studentModel.pickle", "wb") as f:
            pickle.dump(linear, f)

"""

pickle_in = open("Follow_along\studentModel.pickle", "rb")
linear = pickle.load(pickle_in) 

print("Coeffient: ", linear.coef_)
print('intercept: ', linear.intercept_)

"""

#Shows how accurate each result is
predictions = linear.predict(x_test)
for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

"""

#show a graph for p where p is the variable in the data set
p = 'G2'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()

    
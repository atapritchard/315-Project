import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from os import listdir
from os.path import isfile, join
import os

from vectorizer import vectorize

mypath = join(os.getcwd() + "/task1/train")
pos_reviews = [f for f in listdir(mypath + "/positive") if isfile(join(mypath+ "/positive", f))]
neg_reviews = [f for f in listdir(mypath + "/negative") if isfile(join(mypath+ "/negative", f))]

##check if npy file already exists, otherwise gather data
if(os.path.exists("x_train.npy")):
    x_train, y_train = np.load("x_train.npy"), np.load("y_train.npy")
else:
    x_train = np.zeros((len(pos_reviews) + len(neg_reviews), 5000))
    for i in range(len(pos_reviews)):
        x_train[i] = vectorize(mypath+ "/positive" + "/" + pos_reviews[i])
    for j in range(len(neg_reviews)):
        x_train[len(pos_reviews) + j] = vectorize(mypath+ "/positive" + "/" +neg_reviews[j])
    y_train = np.zeros(len(pos_reviews) + len(neg_reviews))
    y_train[:len(pos_reviews)] = 1
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)

reg = LogisticRegressionCV()
reg.fit(x_train, y_train)
print(reg.get_params())

##get test data
if(os.path.exists("x_test.npy")):
    x_test = np.load("x_test.npy")
else:
    path = os.getcwd() + "/task1/test"
    reviews = [f for f in listdir(path) if isfile(join(path, f))]
    x_test = np.zeros((len(reviews), 5000))
    for i in range(len(reviews)):
        x_test[i] = vectorize(path + "/" + reviews[i])
    
    np.save("x_test.npy", x_test)

reg.predict(x_test)



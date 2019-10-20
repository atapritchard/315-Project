import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from os import listdir
from os.path import isfile, join
import os

from vectorizer import vectorize

mypath = "../data"
pos_reviews = [f for f in listdir(mypath + "/pos_reviews") if isfile (join(mypath, f))]
neg_reviews = [f for f in listdir(mypath + "/neg_reviews") if isfile (join(mypath, f))]

if(os.path.exists("x_train.npy")):
    x_train, y_train = np.load("x_train.npy"), np.load("y_train.npy")
else:
    x_train = np.zeros(len(pos_reviews) + len(neg_reviews), 5000)
    for i in range(len(pos_reviews)):
        x_train[i] = vectorize(pos_reviews[i])
    for j in range(len(neg_reviews)):
        x_train[len(pos_reviews) + j] = vectorize(neg_reviews[j])
    y_train = np.zeros(len(pos_reviews) + len(neg_reviews))
    y_train[:len(pos_reviews)] = 1
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)

reg = LogisticRegressionCV()
reg.fit(x_train, y_train)

x_test = np.load("x_test.npy")
reg.predict(x_test)



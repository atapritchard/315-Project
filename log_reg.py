import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from os import listdir
from os.path import isfile, join
import os

from vectorizer import vectorize, get_word_bag
from tqdm import tqdm as tqdm

dtype = np.float16
word_bag = get_word_bag()
mypath = join(os.getcwd() + "/task1/train")
pos_reviews = [f for f in listdir(mypath + "/positive") if isfile(join(mypath+ "/positive", f))]
neg_reviews = [f for f in listdir(mypath + "/negative") if isfile(join(mypath+ "/negative", f))]

##check if npy file already exists, otherwise gather data
if(os.path.exists("x_train.npy")):
    x_train, y_train = np.load("x_train.npy"), np.load("y_train.npy")
    for row in x_train:
        assert np.nan not in row
        assert np.inf not in row
    # print(np.where(x_train >= np.finfo(np.float64).max))
    # exit(0)
else:
    x_train = np.zeros((len(pos_reviews) + len(neg_reviews), 5000), dtype=dtype)
    for i in tqdm(range(len(pos_reviews)), desc='+', ncols=80):
        x_train[i] = vectorize(mypath+ "/positive" + "/" + pos_reviews[i], word_bag)
    for j in tqdm(range(len(neg_reviews)), desc='-', ncols=80):
        x_train[len(pos_reviews) + j] = vectorize(mypath+ "/positive" + "/" +neg_reviews[j], word_bag)
    y_train = np.zeros(len(pos_reviews) + len(neg_reviews), dtype=dtype)
    y_train[:len(pos_reviews)] = 1
    np.save("x_train.npy", np.nan_to_num(x_train))
    np.save("y_train.npy", y_train)

reg = LogisticRegressionCV()
reg.fit(x_train, y_train)
print(x_train[:5].tolist())
print(reg.get_params())

##get test data
if(os.path.exists("x_test.npy")):
    x_test = np.load("x_test.npy")
else:
    path = os.getcwd() + "/task1/test"
    reviews = [f for f in listdir(path) if isfile(join(path, f))]
    x_test = np.zeros((len(reviews), 5000), dtype=dtype)
    for i in range(len(reviews)):
        x_test[i] = vectorize(path + "/" + reviews[i], word_bag)
    
    np.save("x_test.npy", x_test)

preds = reg.predict(x_test)
with open('test_predictions.txt', 'w') as file:
    for line in preds:
        file.write(str(line) + '\n')
print('Predict job completed.')


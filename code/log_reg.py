import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import os

from vectorizer import vectorize, get_word_bag
from tqdm import tqdm as tqdm

dtype = np.float16
word_bag = get_word_bag()
os.chdir('..')
mypath = join(os.getcwd() + "/task1/train")
pos_reviews = [f for f in listdir(mypath + "/positive") if isfile(join(mypath+ "/positive", f))]
neg_reviews = [f for f in listdir(mypath + "/negative") if isfile(join(mypath+ "/negative", f))]

#check if npy file already exists, otherwise gather data
if(os.path.exists("x_train.npy")):
    x, y = np.load("x_train.npy"), np.load("y_train.npy")
    for row in x:
        assert np.nan not in row
        assert np.inf not in row
    # print(np.where(x >= np.finfo(np.float64).max))
    # exit(0)
else:
    x_train = np.zeros((len(pos_reviews) + len(neg_reviews), 5000), dtype=dtype)
    for i in tqdm(range(len(pos_reviews)), desc='+', ncols=80):
        x_train[i] = vectorize(mypath+ "/positive" + "/" + pos_reviews[i], word_bag)
    for j in tqdm(range(len(neg_reviews)), desc='-', ncols=80):
        x_train[len(pos_reviews) + j] = vectorize(mypath+ "/negative" + "/" +neg_reviews[j], word_bag)

y = np.zeros(len(pos_reviews) + len(neg_reviews), dtype=dtype)
y[:len(pos_reviews)] = 1
np.save("x_train.npy", np.nan_to_num(x))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# reg = LogisticRegressionCV(max_iter=100)
# reg.fit(X_train, y_train)
# labs = reg.predict(X_test)
# np.save("x_preds.npy", labs)
# print("accuracy of LR: %i", (sum(np.equal(labs, y_test))/len(y_test)))

nb_reg = GaussianNB()
nb_reg.fit(X_train, y_train)
print("accuracy of NB: %i", (sum(np.equal(nb_reg.predict(X_test), y_test))/len(y_test)))

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
with open('task1_predictions.txt', 'w') as file:
    for line in preds:
        file.write(str(int(line)) + '\n')
print('Predict job completed.')


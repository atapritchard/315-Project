import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import os

from vectorizer import vectorize, get_word_bag
from tqdm import tqdm as tqdm

BAG_SIZE = 5000

dtype = np.float32
word_bag = get_word_bag()
mypath = "../task1/train"
pos_reviews = list(map(lambda x: "{0}.txt".format(x + 1), range(12500)))
neg_reviews = list(map(lambda x: "{0}.txt".format(x + 1), range(12500)))

#check if npy file already exists, otherwise gather data
if(os.path.exists("x_train_small.npy")):
    x = np.load("x_train_small.npy")
    for row in x:
        assert np.nan not in row
        assert np.inf not in row
    # print(np.where(x >= np.finfo(np.float64).max))
    # exit(0)
else:
    x = np.zeros((len(pos_reviews) + len(neg_reviews), BAG_SIZE), dtype=dtype)
    for i in tqdm(range(len(pos_reviews)), desc='+', ncols=80):
        x[i] = vectorize(mypath+ "/positive" + "/" + pos_reviews[i], word_bag)
    for j in tqdm(range(len(neg_reviews)), desc='-', ncols=80):
        x[len(pos_reviews) + j] = vectorize(mypath+ "/negative" + "/" +neg_reviews[j], word_bag)

y = np.zeros(len(pos_reviews) + len(neg_reviews), dtype=dtype)
y[:len(pos_reviews)] = 1
np.save("x_train_small.npy", np.nan_to_num(x))

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(x, y)
# labs = clf.predict(X_test)
# print("accuracy of MLP: %i", (sum(np.equal(labs, y_test))/len(y_test)))

reg = LogisticRegression(C = .001)
reg.fit(x, y)

# nb_reg = GaussianNB()
# nb_reg.fit(X_train, y_train)
# print("accuracy of NB: %i", (sum(np.equal(nb_reg.predict(X_test), y_test))/len(y_test)))

##get test data
if(False):#os.path.exists("x_test.npy")):
    x_test = np.load("x_test.npy")
else:
    path = "../task2_data/test"
    reviews = list(map(lambda x: "{0}.txt".format(x + 1), range(2440)))
    x_test = np.zeros((len(reviews), BAG_SIZE), dtype=dtype)
    for i in range(len(reviews)):
        x_test[i] = vectorize(join(path,reviews[i]), word_bag)
    
    np.save("x_test_2.npy", x_test)

preds = reg.predict(x_test)
print(preds)
with open('task2_predictions.txt', 'w') as file:
    for line in preds:
        file.write(str(int(line)) + '\n')
print('Predict job completed.')


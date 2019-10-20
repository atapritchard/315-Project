import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

x_train, y_train = np.load("x_train.npy"), np.load("y_train.npy")

reg = LogisticRegressionCV()
reg.fit(x_train, y_train)

x_test = np.load("x_test.npy")
reg.predict(x_test)



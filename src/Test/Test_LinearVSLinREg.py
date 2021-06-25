import pickle as pk
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from Linear import *
from Losses import *

if __name__=='__main__':
    #############################SCIKIT LEARN Module#############################
    X, Y = make_regression(n_samples=100, n_features=1, noise=30)
    clf = lm.LinearRegression().fit(X, Y)
    b = clf.intercept_
    w1 = clf.coef_.T
    plt.scatter(X, Y)
    plt.plot(X, w1 * X + b, color='black', label='With scikit')
    plt.legend()
    plt.show()
    #############################LINEAR MODULE#############################
    model = Linear(1, 1)
    loss = MSELoss()
    for _ in range(10000):
        model.zero_grad()
        res1 = model.forward(X)
        resLoss = loss.forward(Y.reshape(-1, 1), res1)
        delta = loss.backward(Y.reshape(-1, 1), res1)
        model.backward_update_gradient(X, delta)
        model.update_parameters()
    plt.scatter(X, Y)
    plt.plot(X, model._parameters[0] * X + model._bias, color='black', label='With Our Module Iter=10000')
    plt.legend()
    plt.show()
import pickle as pk
import numpy as np
import sklearn.linear_model as lm
from mltools import *
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from Linear import *
from Activation import *
from Losses import *
from Sequentiel import *
from Optimizer import *

def test_classification():
    datax, datay = gen_arti(data_type=0, sigma=0.15)
    lin1 = Linear(2, 4, bias=True)
    lin2 = Linear(4, 1, bias=True)
    seq = Sequential(lin1, TanH(), lin2, Sigmoide())
    optim = SGD(seq, MSELoss(), datax, datay, maxIter=100, batch_size=10)
    optim.update()

    def predict(x):
        return np.where(seq.forward(x) >= 0.5, 1, -1)

    plot_frontiere(datax, predict, step=50)
    plot_data(datax, datay.reshape(-1))
    plt.show()

def test_XOR():
    datax, datay = gen_arti(data_type=1, sigma=0.15)
    lin1 = Linear(2, 4, bias=True)
    lin2 = Linear(4, 1, bias=True)
    seq = Sequential(lin1, TanH(), lin2, Sigmoide())
    optim = SGD(seq, MSELoss(), datax, datay, maxIter=500, batch_size=10)
    optim.update()

    def predict(x):
        return np.where(seq.forward(x) >= 0.5, 1, -1)

    plot_frontiere(datax, predict)
    plot_data(datax, datay.reshape(-1))
    plt.show()

def test_XOR_profond():
    datax, datay = gen_arti(data_type=1, sigma=0.15)
    lin1 = Linear(2, 4, bias=True)
    lin2 = Linear(4, 4, bias=True)
    lin3 = Linear(4, 1, bias=True)
    seq = Sequential(lin1, TanH(), lin2, Sigmoide(), lin3, TanH())
    optim = SGD(seq, MSELoss(), datax, datay, maxIter=200, batch_size=10)
    optim.update()

    def predict(x):
        return np.where(seq.forward(x) <= 0.5, -1, 1)

    plot_frontiere(datax, predict)
    plot_data(datax, datay.reshape(-1))
    plt.show()

def test_XOR_more():
    datax, datay = gen_arti(data_type=1, sigma=0.15)
    lin1 = Linear(2, 50, bias=True)
    lin2 = Linear(50, 50, bias=True)
    lin3 = Linear(50, 1, bias=True)
    seq = Sequential(lin1, TanH(), lin2, Sigmoide(), lin3, TanH())
    optim = SGD(seq, MSELoss(), datax, datay, maxIter=20, batch_size=10)
    optim.update()

    def predict(x):
        return np.where(seq.forward(x) <= 0.5, -1, 1)

    plot_frontiere(datax, predict)
    plot_data(datax, datay.reshape(-1))
    plt.show()


def test_Echequier():
    datax, datay = gen_arti(data_type=2, sigma=0.15)
    lin1 = Linear(2, 128, bias=True)
    lin2 = Linear(128, 64, bias=True)
    lin3 = Linear(64, 1, bias=True)
    seq = Sequential(lin1, TanH(), lin2, Sigmoide(), lin3, TanH())
    optim = SGD(seq, MSELoss(), datax, datay, maxIter=500, batch_size=10)
    optim.update()

    def predict(x):
        return np.where(seq.forward(x) <= 0.5, -1, 1)

    plot_frontiere(datax, predict)
    plot_data(datax, datay.reshape(-1))
    plt.show()

if __name__=='__main__':
    test_Echequier()


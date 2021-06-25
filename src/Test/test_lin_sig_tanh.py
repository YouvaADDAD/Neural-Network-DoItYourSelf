from matplotlib import pyplot as plt

from Losses import *
from Linear import *
from Activation import *
import numpy as np
from mltools import gen_arti, plot_frontiere, plot_data

def test_classification():
    datax,datay=gen_arti()
    linear1 =Linear(2,2)
    tan=TanH()
    linear2=Linear(2,1)
    sigmoide=Sigmoide()
    mse = MSELoss()
    for _ in range(100):
        linear1.zero_grad()
        linear2.zero_grad()
        res_lin1 = linear1.forward(datax)
        res_tan=tan.forward(res_lin1)
        res_lin2=linear2.forward(res_tan)
        res_sig=sigmoide.forward(res_lin2)
        res_mse = mse.forward(datay.reshape(-1,1), res_sig)
        delta_mse = mse.backward(datay.reshape(-1,1),res_sig)
        delta_sig = sigmoide.backward_delta(res_lin2,delta_mse)
        linear2.backward_update_gradient(res_tan,delta_sig)
        linear2.update_parameters()
        delta_lin2= linear2.backward_delta(res_tan,delta_sig)
        delta_tanh=tan.backward_delta(res_lin1,delta_lin2)
        linear1.backward_update_gradient(datax,delta_tanh)
        linear1.update_parameters()
        delta_lin1=linear1.backward_delta(datax,delta_tanh)

    def predict(X):
        return np.where(sigmoide.forward(linear2.forward(tan.forward(linear1.forward(X))))>=0.5,1,-1)

    plot_frontiere(datax, predict)
    plot_data(datax, datay.reshape(-1))
    plt.show()

def test_XOR():
    datax, datay = gen_arti(data_type=1)
    linear1 = Linear(2, 4)
    tan = TanH()
    linear2 = Linear(4, 1)
    sigmoide = Sigmoide()
    mse = MSELoss()
    for _ in range(25000):
        linear1.zero_grad()
        linear2.zero_grad()
        res_lin1 = linear1.forward(datax)
        res_tan = tan.forward(res_lin1)
        res_lin2 = linear2.forward(res_tan)
        res_sig = sigmoide.forward(res_lin2)
        res_mse = mse.forward(datay.reshape(-1, 1), res_sig)
        delta_mse = mse.backward(datay.reshape(-1, 1), res_sig)
        delta_sig = sigmoide.backward_delta(res_lin2, delta_mse)
        linear2.backward_update_gradient(res_tan, delta_sig)
        linear2.update_parameters()
        delta_lin2 = linear2.backward_delta(res_tan, delta_sig)
        delta_tanh = tan.backward_delta(res_lin1, delta_lin2)
        linear1.backward_update_gradient(datax, delta_tanh)
        linear1.update_parameters()
        delta_lin1 = linear1.backward_delta(datax, delta_tanh)

        def predict(X):
            return np.where(sigmoide.forward(linear2.forward(tan.forward(linear1.forward(X)))) >= 0.5, 1, -1)
    plot_frontiere(datax, predict, step=100)
    plot_data(datax, datay.reshape(-1))
    plt.show()

if __name__=='__main__':
    test_XOR()
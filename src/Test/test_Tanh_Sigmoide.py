from Losses import *
from Linear import *
from Activation import *
import numpy as np

def optimize():
    np.random.seed(1)
    datax = np.random.randn(20,10)
    datay = np.random.choice([-1,1],20,replace=True)
    linear1 =Linear(10,5)
    tan=TanH()
    linear2=Linear(5,1)
    sigmoide=Sigmoide()
    mse = MSELoss()
    for _ in range(200):
        linear1.zero_grad()
        linear2.zero_grad()
        res_lin1 = linear1.forward(datax)
        res_tan=tan.forward(res_lin1)
        res_lin2=linear2.forward(res_tan)
        res_sig=sigmoide.forward(res_lin2)
        res_mse = mse.forward(datay.reshape(-1,1), res_sig)
        print(np.sum(res_mse))
        delta_mse = mse.backward(datay.reshape(-1,1),res_sig)
        delta_sig = sigmoide.backward_delta(res_lin2,delta_mse)
        linear2.backward_update_gradient(res_tan,delta_sig)
        linear2.update_parameters()
        delta_lin2= linear2.backward_delta(res_tan,delta_sig)
        delta_tanh=tan.backward_delta(res_lin1,delta_lin2)
        linear1.backward_update_gradient(datax,delta_tanh)
        linear1.update_parameters()
        delta_lin1=linear1.backward_delta(datax,delta_tanh)

if __name__=='__main__':
    optimize()
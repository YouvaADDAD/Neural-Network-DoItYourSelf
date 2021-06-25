from Losses import *
from Linear import *
import numpy as np

def optimize():
    datax = np.random.randn(20,10)
    datay = np.random.choice([-1,1],20,replace=True)
    linear =Linear(10,1)
    mse = MSELoss()

    for _ in range(200):
        linear.zero_grad()
        res_lin = linear.forward(datax)
        res_mse = mse.forward(datay.reshape(-1,1), res_lin)
        print('Loss :',res_mse.max())
        delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
        linear.backward_update_gradient(datax,delta_mse)
        grad_lin = linear._gradient
        delta_lin = linear.backward_delta(datax,delta_mse)
        linear.update_parameters()

if __name__=='__main__':
    optimize()
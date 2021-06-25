from Losses import *
from Linear import *
import numpy as np
from Sequentiel import *
from Activation import *

def optimize():
    datax = np.random.randn(20,10)
    datay = np.random.choice([-1,1],20,replace=True)
    seq=Sequential(Linear(10,5),TanH(),Linear(5,1),Sigmoide())
    mse = MSELoss()
    for _ in range(1000):
        seq.zero_grad()
        res_seq=seq.forward(datax)
        res_mse = mse.forward(datay.reshape(-1,1), res_seq)
        print(np.sum(res_mse))
        delta=mse.backward(datay.reshape(-1,1),res_seq)
        seq.backward_delta(datax,delta)
        seq.backward_update_gradient(datax,delta)
        seq.update_parameters()

if __name__=='__main__':
    optimize()
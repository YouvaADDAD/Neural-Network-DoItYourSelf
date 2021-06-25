from Sequentiel import *
from Losses import *
from Activation import *
from Linear import *
from Optimizer import SGD

if __name__=='__main__':
    datax = np.random.randn(20,10)
    datay = np.random.choice([-1,1],20,replace=True)
    seq=Sequential(Linear(10,5),TanH(),Linear(5,1),Sigmoide())
    mse = MSELoss()
    optim=SGD(seq,mse,datax,datay.reshape(-1,1),batch_size=10)
    optim.update()
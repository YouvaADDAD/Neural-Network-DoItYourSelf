import numpy as np
from projet_etu import Module

class AvgPool1D(Module):

    def __init__(self, k_size=3, stride=1):
        super(AvgPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """
        Moyenne sur chaque fenetre
        :param X: (batch,length,chan_in)
        :return: (batch,(length-k_size)/stride +1,chan_in)
        """
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self.stride):
            outPut[:,i,:]=np.mean(X[:,i:i+self.k_size,:],axis=1)
        self._forward=outPut
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """

        :param input: (batch,length,chan_in)
        :param delta: (batch,(length-k_size)/stride +1,chan_in)
        :return: (batch,length,chan_in)
        """
        """
        multiplication (k_size,batch,chan_in)*(batch,chan_in) possible
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros(input.shape)
        for i in range(0,size,self.stride):
            res=np.ones((self.k_size,input.shape[0], input.shape[2])) * delta[:, i, :] / self.k_size
            outPut[:,i:i+self.k_size,:]=res.transpose(1,0,2)
        self._delta=outPut
        return self._delta

if __name__=='__main__':
    layer=AvgPool1D(2,1)
    input=np.random.rand(10,5,2)
    res=layer.forward(input)
    print('Forward',res)
    print('backward',layer.backward_delta(input,res))



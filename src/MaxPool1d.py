import numpy as np
from projet_etu import Module

class MaxPool1D(Module):

    def __init__(self, k_size=3, stride=1):
        super(MaxPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """

        :param X: (batch,length,chan_in)
        :return:  (batch,(length-k_size)/stride +1,chan_in)
        """
        """
        outPut[:,i,:]->batch,chan_in
        X[:,i:i+self.k_size,:]->batch,k_size_,chan_in
        np.max(X[:,i:i+self.k_size,:],axis=1)->batch,chan_in
        """
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self.stride):
            outPut[:,i,:]=np.max(X[:,i:i+self.k_size,:],axis=1)
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
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        outPut=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self.stride):
            indexes_argmax = np.argmax(input[:, i:i+self.k_size,:], axis=1) + i
            outPut[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=outPut
        return self._delta





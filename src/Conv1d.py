import numpy as np
from projet_etu import Module

class Conv1D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        """
        init parameters a (k_size,chan_in,chan_out) et si bias alors init bias a (chan_out)
        :param k_size: int
        :param chan_in: int
        :param chan_out: int
        :param stride: int
        :param bias: bool
        """
        super(Conv1D, self).__init__()
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        bound=1 / np.sqrt(chan_in*k_size)
        #self._parameters=2*(np.random.rand(k_size,chan_in,chan_out)-0.5)*1e-1
        self._parameters = np.random.uniform(-bound, bound, (k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.bias = bias
        if(self.bias):
            #self._bias = 2*(np.random.rand(chan_out)-0.5)*1e-1
            self._bias=np.random.uniform(-bound, bound, chan_out)
            self._gradBias = np.zeros((chan_out))

    def zero_grad(self):
        """
        Remet le gradient a zero, et si bias,remet le gradient de bias a zero
        :return: None
        """
        self._gradient=np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
        Calcule le passe forward
        :param X: (batch,length,chan_in)
        :return: (batch, (length-k_size)/stride +1,chan_out)
        """
        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        output=np.array([(X[:, i: i + self.k_size, :].reshape(X.shape[0], -1)) @ (self._parameters.reshape(-1, self.chan_out)) \
                         for i in range(0,size,self.stride)])
        if (self.bias):
            output+=self._bias
        self._forward=output.transpose(1,0,2)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """
        Met a jour les paramétres
        :param gradient_step:
        :return: None
        """
        self._parameters -= gradient_step * self._gradient
        if self.bias:
            self._bias -= gradient_step * self._gradBias

    def backward_update_gradient(self, input, delta):
        """
        met a jour le gradient
        Calcule le gradient par rapport
        :param input: (batch,length,chan_in)
        :param delta: (batch, (length-k_size)/stride +1,chan_out)
        :return: None
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        output = np.array([ (delta[:,i,:].T) @ (input[:, i: i + self.k_size, :].reshape(input.shape[0], -1))  \
                           for i in range(0, size, self.stride)])
        self._gradient=np.sum(output,axis=0).T.reshape(self._gradient.shape)/delta.shape[0]

        if self.bias:
            self._gradBias=delta.mean((0,1))

    def backward_delta(self, input, delta):
        """
        Calcule du delta
        :param input: (batch,length,chan_in)
        :param delta: (batch, (length-k_size)/stride +1,chan_out)
        :return: (batch,length,chan_in)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        outPut = np.zeros(input.shape)
        for i in range(0, size, self.stride):
            outPut[:,i:i+self.k_size,:] += ((delta[:, i, :]) @ (self._parameters.reshape(-1,self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)
        self._delta= outPut
        return self._delta







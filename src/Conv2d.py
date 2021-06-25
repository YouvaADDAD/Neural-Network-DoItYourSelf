import numpy as np
from projet_etu import Module

class Conv2D(Module):

    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        """
        init parameters a (k_size,k_size,chan_in,chan_out) et si bias alors init bias a (chan_out)
        :param k_size: int
        :param chan_in: int
        :param chan_out: int
        :param stride: int
        :param bias: bool
        """
        super(Conv2D, self).__init__()
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        bound=1 / np.sqrt(chan_in*k_size)
        #self._parameters=2*(np.random.rand(k_size,k_size,chan_in,chan_out)-0.5)*1e-1
        self._parameters = np.random.uniform(-bound, bound, (k_size,k_size,chan_in,chan_out))
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
        self._gradient = np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """

        :param X:(batch, H, W, chan_in)
        :return:(batch,size_H,size_W,chan_out)
        """

        size_h = ((X.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((X.shape[2] - self.k_size) // self.stride) + 1
        outPut=np.zeros((X.shape[0],size_h,size_w,self.chan_out))

        for i in range(0,size_h,self.stride):
            for j in range(0,size_w,self.stride):
                outPut[:,i,j,:]=X[:,i: i + self.k_size,j: j + self.k_size,:].reshape(X.shape[0],-1) @ self._parameters.reshape(-1,self.chan_out)
        if (self.bias):
            outPut += self._bias
        self._forward = outPut
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

        :param input: (batch, H, W, chan_in)
        :param delta: (batch,size_H,size_W,chan_out)
        :attribut _gradient : (k_size,k_size,chan_in,chan_out)
        :return: None mets a jour les gradients
        """
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        outPut=np.zeros(self._gradient.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                    res=(delta[:, i,j, :].T) @ (input[:, i: i + self.k_size,j: j + self.k_size, :].reshape(input.shape[0], -1))
                    outPut+=res.reshape(res.shape[0],self.k_size,self.k_size,input.shape[-1]).transpose(1,2,3,0)
        self._gradient=outPut/delta.shape[0]
        if self.bias:
            self._gradBias=delta.mean((0,1,2))

    def backward_delta(self, input, delta):
        """

        :param input: (batch, H, W, chan_in)
        :param delta: (batch,size_H,size_W,chan_out)
        :return: (batch, H, W, chan_in)
        """
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        outPut = np.zeros(input.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                res=(delta[:,i,j,:])@(self._parameters.reshape(-1,self.chan_out).T)
                #Res->(batch,k_size*k_size*chan_in)
                outPut[:, i:i + self.k_size, j:j + self.k_size, :]+=res.reshape(delta.shape[0],self.k_size,self.k_size,self.chan_in)
                #outPut[:,i:i+self.k_size,j:j+self.k_size,:]->batch,k_size,k_size,chan_in
                #delta[:,i,j,:]->batch,chan_out
                #parameters->(k_size,k_size,chan_in,chan_out)
                #batch,k_size,k_size,chan_in
        self._delta=outPut
        return self._delta





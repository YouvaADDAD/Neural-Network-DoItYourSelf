import numpy as np
from projet_etu import Module

class Linear(Module):
    def __init__(self, input, output, bias=True):
        """

        :param input: int (input size)
        :param output: int (output size)
        :param bias: boolean
        """
        super(Linear, self).__init__()
        self.input=input
        self.output=output
        #bound=1 / np.sqrt(input)
        self._parameters = 2 * (np.random.rand(input, output) - 0.5)
        #self._parameters = np.random.uniform(-bound, bound, (input,output))
        self._gradient = np.zeros((input, output))
        self.bias = bias
        if (self.bias):
            self._bias = 2 * (np.random.randn(output) - 0.5)
            #self._bias=np.random.uniform(-bound, bound, output)
            self._gradBias = np.zeros(output)

    def zero_grad(self):
        """
        Reset gradient and if bias equal to true reset gradBias
        :return: None
        """
        self._gradient = np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
        campute linear forward
        :param X: batch*input
        :return: batch*output
        """
        self._forward=np.dot(X,self._parameters)
        if(self.bias):
                self._forward = np.add(self._forward,self._bias)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """
        update parameters and if bias is true ,update bias
        :param gradient_step:
        :return: None
        """
        self._parameters -= gradient_step*self._gradient
        if(self.bias):
            self._bias -= gradient_step*self._gradBias

    def backward_update_gradient(self, input, delta):
        """
        met a jour les gradients ,si bias alors il mets a jour le gradient du bias
        :param input: batch*input
        :param delta: batch*output
        :return: None
        """
        self._gradient = np.dot(input.T, delta)/delta.shape[0]
        if (self.bias):
            self._gradBias = np.sum(delta,axis=0)

    def backward_delta(self, input, delta):
        """
        compute delta
        :param input: batch * input
        :param delta:batch * output
        :return: batch*input
        """
        self._delta= np.dot(delta,self._parameters.T)
        return self._delta

    def __str__(self):
        return f'Linear Modular with input :{self.input} and output : {self.output}'
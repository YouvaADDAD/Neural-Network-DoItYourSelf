import numpy as np
from projet_etu import Module


class TanH(Module):
    def __init__(self):
        super(TanH,self).__init__()

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        """
        calcul le forward on appliquant une activation Tanh
        :param X: batch*input
        :return: batch*output
        """
        self._forward=np.tanh(X)
        return self._forward


    def derivate_TanH(self,input):
        """
        calcule la derivée de la tanh
        :param input:
        :return: 1-tanh(input)^2
        On aura pu directement utiliser le calcule du forward
        """
        return 1-(np.tanh(input)**2)

    def backward_delta(self, input, delta):
        """
        :param input:
        :param delta:
        :return: le delta de taille batch*input ->input=output ici
        """
        ## Calcul la derivee de l'erreur
        assert (input.shape == delta.shape)
        self._delta = np.multiply(delta, self.derivate_TanH(input))
        return self._delta

    def __str__(self):
        return 'Tangente Hyperbolic activation'

class Sigmoide(Module):

    def __init__(self):
        super(Sigmoide, self).__init__()

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        """
        calcul le forward on appliquant une activation Sigmoide
        :param X: batch*input
        :return: batch*output
        """
        self._forward =self.sigmoide(X)
        return self._forward

    def sigmoide(self,input):
        return  1 / (1 + np.exp(-input))

    def derivate_Sigmoide(self,input):
        """
        :param input:
        :return: derivative of sigmoid per rapport to input
        """
        return self.sigmoide(input) * (1 - self.sigmoide(input))

    def backward_delta(self, input, delta):
        """
        calcule du delta par rapport aux input
        :param input:
        :param delta:
        :return: delta de taille batch*input (input=output ici)
        """
        assert (input.shape == delta.shape)
        self._delta= np.multiply(delta , self.derivate_Sigmoide(input))
        return self._delta

    def __str__(self):
        return 'Sigmoide Activation'

class SoftMax(Module):
    def __init__(self):
        super(SoftMax,self).__init__()

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        self._forward=self.softmax(X)
        return self._forward


    def softmax(self,input):
        """
        :param input: batch*input
        :return: batch*input
        """
        input = input - np.max(input, axis=1,keepdims=True)
        sum_exp = np.sum(np.exp(input), axis=1,keepdims=True)
        return np.exp(input)/sum_exp

    def derivative_softMax(self,input):
        return np.multiply(self.softmax(input), (1 - self.softmax(input)))

    def backward_delta(self, input, delta):
        """
        calcule du delta par rapport aux input ,ici c'est le delta qui annule les entrées qui rentre pas dans propagation
        :param input:
        :param delta:
        :return: delta de taille batch*input (input=output ici)
        """
        assert (input.shape == delta.shape)
        self._delta = np.multiply(delta, self.derivative_softMax(input))
        return self._delta

    def __str__(self):
        return 'SoftMax Activation'


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax,self).__init__()

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        maxInput=np.max(input, axis=1,keepdims=True)
        logSumExp=np.log(np.exp(X-maxInput).sum(axis=1,keepdims=True))
        self._forward=X-maxInput-logSumExp
        return self._forward

    def backward_delta(self, input, delta):
        """
        calcule du delta par rapport aux input ,ici c'est le delta qui annule les entrées qui rentre pas dans propagation
        :param input:
        :param delta:
        :return: delta de taille batch*input (input=output ici)
        """
        assert (input.shape == delta.shape)
        maxInput=np.max(input, axis=1,keepdims=True)
        self._delta = np.multiply(1 - np.exp(input-maxInput)/np.exp(input-maxInput).sum(axis=1,keepdims=True), delta)
        return self._delta

    def __str__(self):
        return 'LogSoftMax Activation'


class Threshold(Module):
    def __init__(self,threshold):
        """
        :param threshold: float pour la valeur du seuil
        """
        super(Threshold,self).__init__()
        self._threshold=threshold

    def update_parameters(self, gradient_step=1e-3):
        pass

    def forward(self, X):
        self._forward=self.threshold(X)
        return self._forward

    def threshold(self,input):
        return np.where(input>self._threshold,input,0.)


    def derivative_Threshold(self,input):
        #Batch x Output
        #np.where(self.threshold(input)<=self._threshold,0.,1.)
        return (input > self._threshold).astype(float)

    def backward_delta(self, input, delta):
        self._delta=np.multiply(delta,self.derivative_Threshold(input))
        return self._delta

    def __str__(self):
        return 'Threshold Activation'

class ReLU(Threshold):
    def __init__(self):
        super(ReLU,self).__init__(0.)

    def __str__(self):
        return 'ReLU Activation'





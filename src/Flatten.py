from projet_etu import Module

class Flatten(Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        """
        Calcule la passe forward
        :param X:(batch,length,chan_in)
        :return:(batch,length*chan_in)
        """
        self._forward = X.reshape(X.shape[0], -1)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
        calcule le pass backward
        :param input: (batch,length,chan_in)
        :param delta: (batch, length * chan_in)
        :return: (batch,length,chan_in)
        """
        self._delta = delta.reshape(input.shape)
        return self._delta

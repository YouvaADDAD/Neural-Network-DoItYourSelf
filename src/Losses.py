import numpy as np
from Activation import LogSoftMax

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass

class MSELoss(Loss):
    """
            MSE loss
        y: batch*dim_out true value
        yhat: batch*dim_out predicted value
        forward :return batch*1
        backward:return batch*dim_out
    """
    def forward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return np.sum((y-yhat)**2,axis=1,keepdims=True)

    def backward(self, y, yhat):
        assert (y.shape == yhat.shape)
        return 2*(yhat-y)

    def __str__(self):
        return 'MSELoss'

class BCELoss(Loss):
    """
    Binary Cross Enropy
    y: batch*dim_out with label binary (0,1);(1,-1)
    yhat: batch*dim_out
    forward :return batch*1
    backward:return batch*dim_out

    Le sum on sais jamais on le laisse comme sa parce que a priori le y et yhat sont de dimension batch*1
    yhat->probabilité ou vecteur one hot
    y->true label
    """

    def forward(self, y, yhat):
        assert y.shape==yhat.shape
        return np.sum((y - 1) * np.maximum(np.log(1 - yhat), -100) - y * np.maximum(np.log(yhat), -100),axis=1,keepdims=True)

    def backward(self, y, yhat):
        assert y.shape==yhat.shape
        """
        Reformulation de :
        -> -(y-1)/1-yhat -y/yhat
        -> (1-y)/1-yhat -y/yhat
        -> (1-y)*yhat -y(1-yhat)/(1-yhat)yhat
        -> yhat -y*yhat -y +y*yhat / (1-yhat)*yhat
        -> yhat-y/(1-yhat)*yhat
        """
        return (yhat - y)/ (np.maximum((1 - yhat) *yhat, 1e-12))

    def __str__(self):
        return 'Binary Cross Entropy Loss'

class CrossEntropyLoss(Loss):
    """
    CrossEntropy
     y: batch*dim_out with label multiclasse
    yhat: batch*dim_out
    forward :return batch*1
    backward:return batch*dim_out
    Utilisation directement aprés un softMax
    """
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        #return np.sum(-y * np.log(yhat+1e-10), axis=1,keepdims=True)
        return np.sum(-y * np.maximum(np.log(yhat), -100), axis=1, keepdims=True)

    def backward(self, y, yhat):
        # y->Batch x outPut
        assert y.shape == yhat.shape
        #return -y / (yhat + 1e-10)
        return -y/np.maximum(yhat, 1e-12)

    def __str__(self):
        return 'Cross Entropy Loss'

class NLLLoss(Loss):
    """
    Cross Entropy loss
    y:batch*dim_out (one hot)
    yhat: batch*dim_out
    forward:-yhat de la classe correspondante
    """

    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return np.sum(-y*yhat,axis=1,keepdims=True)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -y

    def __str__(self):
        return 'Cross Entropy Loss'

class CrossEntropyCriterion(Loss):
    
    def __init__(self):
        self.lsm = LogSoftMax()
        self.nll=  NLLLoss()
        self.yhat=None


    def forward(self, y, yhat):
        self.yhat=self.lsm.forward(yhat)
        return self.nll.forward(y,yhat)

    def backward(self, y, yhat):
        delta=self.nll.backward(y,self.yhat)
        return self.lsm.backward_delta(yhat,delta)

    def __str__(self):
        return 'CrossEntropyCriterion'


class CELogSoftMax(Loss):
    """
    CrossEntropy log softmax
    y:batch*dim_out
    yhat:batch*dim*dim_out
    """

    def forward(self, y, yhat):
        assert y.shape==yhat.shape
        return np.sum(-y*yhat,axis=1,keepdims=True)+np.log(np.sum(np.exp(yhat),axis=1,keepdims=True))

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -y+ (y*np.exp(yhat))/np.sum(np.exp(yhat),axis=1,keepdims=True)

    def __str__(self):
        return 'Cross Entropy LogSoftMax Loss'


class HingeLoss(Loss):
    """
        Hinge Loss
         y: batch*dim_out with label (-1,1)
        yhat: batch*dim_out
        forward :return batch*1
        backward:return batch*dim_out

        """
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return np.sum(np.maximum(1. - y * yhat, 0.), axis=1,keepdims=True)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        forw = np.maximum(1. - y * yhat, 0.)
        return np.where(forw > 0, -y, 0)

    def __str__(self):
        return 'Hinge Loss'

class ABSLoss(Loss):
    """
    ABSLoss
    y: batch*out_out
    yhat:batch*output
    forward :return batch*1
    backward:return batch*dim_out
    """
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        return np.sum(np.abs(y-yhat), axis=1,keepdims=True)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        forw = y-yhat
        return np.where(forw > 0, -1, 1)

    def __str__(self):
        return 'Absolute Loss'







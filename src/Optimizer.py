import numpy as np
class Optim(object):
    def __init__(self,net, loss, eps=1e-3):
        """
        Constructor
        :param net: Module
        :param loss: Loss
        :param eps: float
        """
        self.net=net
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
        """
        Applique une itération de la descente de gradient
        :param batch_x: batch*input
        :param batch_y: batch*output
        :return:None
        """
        res_net=self.net.forward(batch_x)
        res_loss=self.loss.forward(batch_y,res_net)
        #print('Loss :' ,res_loss.sum())
        delta_loss=self.loss.backward(batch_y,res_net)
        self.net.backward_delta(batch_x,delta_loss)
        self.net.backward_update_gradient(batch_x,delta_loss)
        self.net.update_parameters(self.eps)
        self.net.zero_grad()
        #return res_loss.sum()#ajout

    def update(self):
        pass


class SGD(Optim):
    def __init__(self, net, loss,datax,datay,batch_size=10,maxIter=200, eps=1e-3):
        """
        Applique un mini batch
        :param net: Module
        :param loss: Loss
        :param datax: Batch*input
        :param datay: Batch*output
        :param batch_size: int
        :param maxIter: int
        :param eps: float
        """
        super(SGD,self).__init__(net, loss, eps)
        assert datax.shape[0]==datay.shape[0]
        self.datax=datax
        self.datay=datay
        self.batch_size=batch_size
        self.maxIter=maxIter

    def generator(self):
        """
        creer un n_split de batch_x et de batch_y
        :return: renvoie une générateur sur les batch's
        """
        length=len(self.datax)
        indices = np.arange(length)
        np.random.shuffle(indices)
        n_split = length // self.batch_size
        if (length % self.batch_size != 0):
            n_split += 1
        for i in range(n_split):
            indexes=indices[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.datax[indexes],self.datay[indexes]

    def update(self):
        """
        Mise a jour global du réseau
        :return: None
        """
        #moyenne=[]
        for _ in range(self.maxIter):
            #res=[]
            for batch_x,batch_y in self.generator():
                assert(batch_x.shape[0]==batch_y.shape[0])
                self.step(batch_x,batch_y)
                #res.append(self.step(batch_x,batch_y))
            #moyenne.append(np.mean(res))
            #print(np.mean(res))
        #return moyenne





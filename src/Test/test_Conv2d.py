from Linear import *
from Sequentiel import *
from Activation import TanH,Sigmoide,ReLU,SoftMax
from Losses import BCELoss,MSELoss,ABSLoss,CELogSoftMax,CrossEntropyLoss
from Optimizer import SGD
import matplotlib.pylab as plt
from mltools import *
from Conv2d import *
from Flatten import *
from MaxPool2d import *
from sklearn.metrics import confusion_matrix
import numpy as np

def OneHotEncoding(y):
    onehot = np.zeros((y.size,y.max()+1));
    onehot[np.arange(y.size),y]=1
    return onehot


def test_Conv2d():
    #Load Data From USPS , directement pris depuis TME4
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    


    #taille couche
    alltrainy_oneHot = OneHotEncoding(alltrainy)
    alltesty_oneHot = OneHotEncoding(alltesty)
    alltrainx=alltrainx.reshape(alltrainx.shape[0],16,16,1)
    alltestx=alltestx.reshape(alltestx.shape[0],16,16,1)

    iteration = 200
    gradient_step = 1e-3
    batch_size = 200

    l1=Conv2D(3,1,32) 
    l2=MaxPool2D(2,2) 
    l3=Flatten()
    l4=Linear(1568,100) 
    l5=ReLU() 
    l6=Linear(100,10)
    l7=SoftMax()
    
    model=Sequential(l1,l2,l3,l4,l5,l6,l7)
    loss = CrossEntropyLoss()
    opt = SGD(model,loss,alltrainx, alltrainy_oneHot, batch_size, maxIter=iteration,eps=gradient_step)
    opt.update()
    
    #Predection
    predict = model.forward(alltrainx)
    predict = np.argmax(predict, axis=1)
    
    #Confusion Matrix
    confusion = confusion_matrix(predict, alltrainy)
    print(np.sum(np.where(predict==alltrainy,1,0))/len(predict))
    plt.imshow(confusion)



if __name__=='__main__':
    test_Conv2d()






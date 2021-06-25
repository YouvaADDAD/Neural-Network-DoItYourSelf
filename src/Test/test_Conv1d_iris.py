from Linear import *
from Sequentiel import *
from Activation import TanH,Sigmoide,ReLU,SoftMax
from Losses import BCELoss,MSELoss,ABSLoss,CELogSoftMax,CrossEntropyLoss
from Optimizer import SGD
from mltools import load_usps
import matplotlib.pylab as plt
from Conv1d import *
from Conv2d import *
from Flatten import *
from MaxPool1d import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

def OneHotEncoding(y,num_classes):
    onehot = np.zeros((y.size,num_classes));
    onehot[np.arange(y.size),y]=1
    return onehot

def test_conv1d_iris():
    df = pd.read_csv('../data/iris.csv')
    df['labels'] =df['class'].astype('category').cat.codes

    X = df[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
    Y = df['labels']
    x_train, x_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(Y), test_size=0.33, shuffle= True)

    # The known number of output classes.
    num_classes = 3

    # Input image dimensions
    input_shape = (4,)

    # Convert class vectors to binary class matrices. This uses 1 hot encoding.
    y_train_binary = OneHotEncoding(y_train, num_classes)
    y_test_binary = OneHotEncoding(y_test, num_classes)


    #Reshape
    x_train = x_train.reshape(100, 4,1)
    x_test = x_test.reshape(50, 4,1)

    iteration = 200
    gradient_step = 1e-3
    batch_size = 20
    model = Sequential(Conv1D(2,1,16),MaxPool1D(2,2),Flatten(),Linear(16,3),SoftMax() )
    loss = CrossEntropyLoss()
    opt = SGD(model,loss,x_train, y_train_binary, batch_size, maxIter=iteration,eps=gradient_step)
    opt.update()

    predict = model.forward(x_test)
    predict = np.argmax(predict, axis=1)
    confusion = confusion_matrix(predict, y_test)
    print(np.sum(np.where(predict==y_test,1,0))/len(predict))
    plt.imshow(confusion)





if __name__=='__main__':
    test_conv1d_iris()


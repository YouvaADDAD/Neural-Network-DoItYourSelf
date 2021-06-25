from sklearn.metrics import confusion_matrix
from mltools import *
import matplotlib.pyplot as plt
from Linear import *
from Activation import *
from Losses import *
from Sequentiel import *
from Optimizer import *
import numpy as np


def OneHotEncoding(y):
    onehot = np.zeros((y.size, y.max() + 1));
    onehot[np.arange(y.size), y] = 1
    return onehot


def test_multiclass():
    # Load Data From USPS , directement pris depuis TME4
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)

    # taille couche
    input = len(alltrainx[0])
    output = len(np.unique(alltesty))
    alltrainy_oneHot = OneHotEncoding(alltrainy)

    # Hyperparameters
    maxIter = 200
    eps = 1e-3
    batch_size = 100

    linear1 = Linear(input, 128)
    activation1 = TanH()
    linear2 = Linear(128, 64)
    activation2 = TanH()
    linear3 = Linear(64, output)
    activation3 = SoftMax()
    loss = CrossEntropyLoss()

    # Optimization
    model = Sequential(linear1, activation1, linear2, activation2, linear3, activation3)
    optimizer = SGD(model, loss, alltrainx, alltrainy_oneHot, batch_size=batch_size, eps=eps, maxIter=maxIter)
    optimizer.update()

    # Predection
    predict = model.forward(alltrainx)
    predict = np.argmax(predict, axis=1)

    # Confusion Matrix
    confusion = confusion_matrix(predict, alltrainy)
    print(np.sum(np.where(predict == alltrainy, 1, 0)) / len(predict))
    plt.imshow(confusion)


if __name__ == '__main__':
    test_multiclass()
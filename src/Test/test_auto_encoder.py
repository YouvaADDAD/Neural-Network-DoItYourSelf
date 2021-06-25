
import tensorflow as tf
from Linear import *
from Sequentiel import *
from Activation import TanH, Sigmoide
from Losses import BCELoss
from Optimizer import SGD
from mltools import load_usps
import matplotlib.pylab as plt


def test_autoEncoder():
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)

    iteration = 200
    eps = 1e-3
    batch_size = 10
    l1 = Linear(256, 100)
    l2 = Linear(100, 10)
    l3 = Linear(10, 100)
    l4 = Linear(100, 256)
    l3._parameters = l2._parameters.T.copy()
    l4._parameters = l1._parameters.T.copy()

    encoder = Sequential(l1, TanH(), l2, TanH())
    decoder = Sequential(l3, TanH(), l4, Sigmoide())
    model = Sequential(encoder, decoder)
    loss = BCELoss()
    opt = SGD(model, loss, alltrainx, alltrainx, batch_size, maxIter=iteration, eps=eps)
    opt.update()
    predict = model.forward(alltrainx)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(predict[i].reshape(16, 16))
        plt.gray()
    plt.show()
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)

        plt.imshow(alltrainx[i].reshape(16, 16))
        plt.gray()

    plt.show()


def test_autoEncoder_debruitage():
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    noise_factor = 0.2
    x_train_noisy = alltrainx + noise_factor * tf.random.normal(shape=alltrainx.shape).numpy()
    iteration = 200
    eps = 1e-3
    batch_size = 10
    l1 = Linear(256, 100)
    l2 = Linear(100, 10)
    l3 = Linear(10, 100)
    l4 = Linear(100, 256)
    l3._parameters = l2._parameters.T.copy()
    l4._parameters = l1._parameters.T.copy()

    encoder = Sequential(l1, TanH(), l2, TanH())
    decoder = Sequential(l3, TanH(), l4, Sigmoide())
    model = Sequential(encoder, decoder)
    loss = BCELoss()
    opt = SGD(model, loss, x_train_noisy, x_train_noisy, batch_size, maxIter=iteration, eps=eps)
    opt.update()
    predict = model.forward(x_train_noisy)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(predict[i].reshape(16, 16))
        plt.gray()
    plt.show()
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)

        plt.imshow(x_train_noisy[i].reshape(16, 16))
        plt.gray()

    plt.show()


if __name__ == '__main__':
    test_autoEncoder_debruitage()
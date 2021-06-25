from Linear import *
from Sequentiel import *
from Activation import *
from Losses import*
from Optimizer import SGD
from mltools import load_usps
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def test_cluster():

    ########################Lecture des données
    uspsdatatrain = "./data/USPS_train.txt"
    uspsdatatest = "./data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)

    min_val = alltrainx.min()
    max_val = alltrainx.max()
    alltrainx = (alltrainx - min_val) / (max_val - min_val)
    alltestx = (alltestx - min_val) / (max_val - min_val)

    # TSNE aux données
    reduce_data = TSNE(n_components=2).fit_transform(alltrainx)
    target_ids = np.unique(alltrainy)
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_ids):
        plt.scatter(reduce_data[alltrainy == i, 0], reduce_data[alltrainy == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

    #Kmeans
    tsne_train = TSNE(n_components=2).fit_transform(alltrainx)
    kmeans = KMeans(n_clusters=10)
    y_means = kmeans.fit_predict(tsne_train)
    plt.figure(figsize=(6, 5))
    labels = np.unique(y_means)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for label, c in zip(labels, colors):
        plt.scatter(tsne_train[y_means == label, 0], tsne_train[y_means == label, 1], c=c, label=label)
    plt.legend()
    plt.show()

    #Train autoencoder
    iteration = 80
    eps = 1e-3
    batch_size = 10

    l1 = Linear(256, 32)
    l2 = Linear(32, 16)
    l3 = Linear(16, 10)
    l4 = Linear(10, 16)
    l5 = Linear(16, 32)
    l6 = Linear(32, 256)

    encoder = Sequential(l1, TanH(), l2, TanH(), l3, TanH())
    decoder = Sequential(l4, TanH(), l5, TanH(), l6, Sigmoide())
    model = Sequential(encoder, decoder)
    loss = MSELoss()
    opt = SGD(model, loss, alltrainx, alltrainx, batch_size, maxIter=iteration, eps=eps)
    opt.update()

    #Visualisation
    dataX = encoder.forward(alltrainx)
    tsne_train2 = TSNE(n_components=2).fit_transform(dataX)
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    plt.figure(figsize=(6, 5))
    labels = np.unique(alltrainy)
    for label, c in zip(labels, colors):
        plt.scatter(tsne_train2[alltrainy == label, 0], tsne_train2[alltrainy == label, 1], c=c, label=label)
    plt.legend()
    plt.show()


if __name__=='__main__':
    test_cluster()

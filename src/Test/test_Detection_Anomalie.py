from Linear import *
from Sequentiel import *
from Activation import TanH,Sigmoide,ReLU
from Losses import BCELoss,MSELoss,ABSLoss
from Optimizer import SGD
from mltools import load_usps
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix,plot_confusion_matrix

def test_detection_anomalie():
    df=pd.read_csv('../data/ECG5000.csv').drop(columns='Unnamed: 0')
    raw_data = df.values
    labels = raw_data[:, -1].astype(int)
    data = raw_data[:, 0:-1]

    #Test_train_split
    train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21)

    # Normalisation
    min_val = train_data.min()
    max_val = train_data.max()
    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    # Train les rythmes normaux
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anormalous_train_data = train_data[~train_labels]
    anormalous_test_data = test_data[~test_labels]

    #Plot un exemple de rythme normal
    plt.grid()
    plt.plot(np.arange(140), normal_train_data[0])
    plt.title("Rythme normale")
    plt.show()

    #Plot un exemple de rythme anormal
    plt.grid()
    plt.plot(np.arange(140), anormalous_train_data[0])
    plt.title("Rythme anormale")
    plt.show()

    #Hyperparametre
    iteration = 200
    eps = 1e-3
    batch_size = 10

    #L'autoEncoder et optimisation
    l1 = Linear(140, 32)
    l2 = Linear(32, 16)
    l3 = Linear(16, 8)
    l4 = Linear(8, 16)
    l5 = Linear(16, 32)
    l6 = Linear(32, 140)

    encoder = Sequential(l1, TanH(), l2, TanH(), l3, TanH())
    decoder = Sequential(l4, TanH(), l5, TanH(), l6, Sigmoide())
    model = Sequential(encoder, decoder)
    loss = ABSLoss()
    opt = SGD(model, loss, normal_train_data, normal_train_data, batch_size, maxIter=iteration, eps=eps)
    opt.update()

    #Reconstruction normal test
    encoded_imgs = encoder.forward(normal_test_data)
    decoded_imgs = decoder.forward(encoded_imgs)

    plt.plot(normal_test_data[0], 'b')
    plt.plot(decoded_imgs[0], 'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], normal_test_data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

    #Reconstruction anormal test
    encoded_imgs = encoder.forward(anormalous_test_data)
    decoded_imgs = decoder.forward(encoded_imgs)

    plt.plot(anormalous_test_data[0], 'b')
    plt.plot(decoded_imgs[0], 'r')
    plt.fill_between(np.arange(140), decoded_imgs[0], anormalous_test_data[0], color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()

    #Loss recostruction
    reconstructions = model.forward(normal_train_data)
    train_loss = ABSLoss().forward(reconstructions, normal_train_data)

    plt.hist(train_loss.ravel(), bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()

    threshold = np.mean(train_loss) + np.std(train_loss)
    print("Threshold: ", threshold)
    preds = predict(model, test_data, threshold)
    print_stats(preds, test_labels)
    plt.imshow(confusion_matrix(preds, test_labels))
    plt.show()


def predict(model, data, threshold):
    reconstructions = model.forward(data)
    loss = ABSLoss().forward(reconstructions, data)
    return np.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))



if __name__=='__main__':
    test_detection_anomalie()



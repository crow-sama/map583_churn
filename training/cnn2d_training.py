from itertools import product
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
from tqdm import tqdm



def train_epoch(loader, model, loss_fn, optimizer):
    model.train()
    losses = []
    for X, y in tqdm(loader.train()):
        # batch by batch
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def test_epoch(loader, model, loss_fn):
    model.eval()
    losses = []
    accuracy = []
    # confusion matrix
    cm = np.zeros((3, 3))
    for X, y in tqdm(loader.test()):
        pred = model(X)
        classes = torch.argmax(pred, dim=1)

        fill = np.ones(classes.shape[0])
        cm += coo_matrix((fill, (classes.cpu(), y.cpu())), shape=(3, 3)).toarray()

        batch_accuracy = (classes == y).sum().item() / classes.shape[0]
        accuracy.append(batch_accuracy)

        loss = loss_fn(pred, y)
        losses.append(loss.item())

    return np.mean(losses), np.mean(accuracy), cm


def fit(model, loader, optimizer, loss_fn, nb_epochs):
    # fit the model by training it nb_epochs times
    train_loss_t = []
    test_loss_t = []
    for epoch in range(nb_epochs):
        print("Epoch: {}/{}".format(epoch + 1, nb_epochs))
        train_loss = train_epoch(loader, model, loss_fn, optimizer)
        test_loss, test_accuracy, cm = test_epoch(loader, model, loss_fn)
        train_loss_t.append(train_loss)
        test_loss_t.append(test_loss)
        print("Train set: Average loss: {:.4f}\n".format(train_loss))
        print("Test set: Average loss: {:.4f}".format(test_loss, test_accuracy))
        print("          Accuracy: {:.4f}".format(test_accuracy))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("          Class 0 Accuracy: {:.4f}".format(cm[0, 0]))
        print("          Class 1 Accuracy: {:.4f}".format(cm[1, 1]))
        print("          Class 2 Accuracy: {:.4f}\n".format(cm[2, 2]))
        

    return model, train_loss_t, test_loss_t, cm


def plot_losses(train_loss_t, test_loss_t):
    nb_epochs = len(train_loss_t)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(range(nb_epochs), train_loss_t, color='orange', label='Loss on the training set')
    ax.plot(range(nb_epochs), test_loss_t, color='green', label='Loss on the testing set')
    ax.legend()
    return fig


def save_model_and_losses(train_loss_t, test_loss_t, model, model_name):
    filename_train_loss = model_name + '_train'
    filename_test_loss = model_name + '_test'
    np.savetxt(filename_train_loss, train_loss_t)
    np.savetxt(filename_test_loss, test_loss_t)
    torch.save(model.state_dict(), model_name)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontsize=25)
    tick_marks = np.arange(len(classes))
    ax.xaxis.set_ticks(tick_marks, classes)
    ax.yaxis.set_ticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize = 14)

        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

    return fig

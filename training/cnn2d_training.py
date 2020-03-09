from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

import torch



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
    for X, y in tqdm(loader.test()):
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
    return np.mean(losses)


def fit(model, loader, optimizer, loss_fn, nb_epochs):
    # fit the model by training it nb_epochs times
    train_loss_t = []
    test_loss_t = []
    for epoch in range(nb_epochs):
        print("Epoch: {}/{}".format(epoch + 1, nb_epochs))
        train_loss = train_epoch(loader, model, loss_fn, optimizer)
        test_loss = test_epoch(loader, model, loss_fn)
        train_loss_t.append(train_loss)
        test_loss_t.append(test_loss)
        print("Train set: Average loss: {:.4f}\n".format(train_loss))
        print("Test set: Average loss: {:.4f}\n".format(test_loss))
    return model, train_loss_t, test_loss_t


def plot_losses(train_loss_t, test_loss_t):
    nb_epochs = len(train_loss_t)
    plt.plot(range(nb_epochs), train_loss_t, color='orange', label='Loss on the training set')
    plt.plot(range(nb_epochs), test_loss_t, color='green', label='Loss on the testing set')
    plt.legend()
    plt.show()


def save_model_and_losses(train_loss_t, test_loss_t, model, model_name):
    filename_train_loss = model_name + '_train'
    filename_test_loss = model_name + '_test'
    np.savetxt(filename_train_loss, train_loss_t)
    np.savetxt(filename_test_loss, test_loss_t)
    torch.save(model.state_dict(), model_name)

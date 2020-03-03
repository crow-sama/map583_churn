from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch


def overfitting(model, optimizer, loss_fn, X, y, nb_epochs):
    # over fitting on same time series to make sure it works
    for i in range(1, nb_epochs + 1):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            message1 = 'Loop : {}/{}'.format(i, nb_epochs)
            print(message1)
            message2 = 'On sample error : {:.4f}\n'.format(loss.item())
            print(message2)


def train_epoch(X_train, y_train, model, loss_fn, optimizer):
    # train the model through the whole training dataset
    nb_train_sequences = len(y_train)
    model.train()
    losses = []
    for k in tqdm(range(nb_train_sequences)):
        X, y = X_train[k], y_train[k]
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def test_epoch(X_test, y_test, model, loss_fn):
    # evaluate the model through the whole testing dataset
    nb_test_sequences = len(y_test)
    model.eval()
    losses = []
    for k in range(nb_test_sequences):
        X, y = X_test[k], y_test[k]
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
    return np.mean(losses)


def fit(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, nb_epochs):
    # fit the model by training it nb_epochs times
    train_loss_t = []
    test_loss_t = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1, verbose='True',
                                                           threshold=0.001)
    for epoch in range(0, nb_epochs):
        message1 = 'Epoch: {}/{}'.format(epoch + 1, nb_epochs)
        print(message1)
        train_loss = train_epoch(X_train, y_train, model, loss_fn, optimizer)
        test_loss = test_epoch(X_test, y_test, model, loss_fn)
        message2 = 'Train set: Average loss: {:.4f}\n'.format(train_loss)
        message3 = 'Test set: Average loss: {:.4f}\n'.format(test_loss)
        scheduler.step(test_loss)
        train_loss_t.append(train_loss)
        test_loss_t.append(test_loss)
        print(message2)
        print(message3)
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

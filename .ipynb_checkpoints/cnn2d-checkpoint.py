from preprocessing import cnn2d_preprocessing as cnn_pre
from models import cnn2d_models as cnn_mod
from training import cnn2d_training as cnn_trn 

import git
import torch
import torch.nn as nn
import os


# import data
repo_not_cloned = not(os.path.isdir('RUL-Net'))
if repo_not_cloned:
    git.Git().clone("https://github.com/LahiruJayasinghe/RUL-Net.git")


# remove *.npy files when you change the parameters
loader = cnn_pre.Loader(use_cuda=True, batch_size=50, w0=60, w1=100)

model = cnn_mod.Net(num_channels=loader.num_channels)

if loader.cuda:
    model = model.cuda()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

nb_epochs = 20

model_name = "CNN2d"

model, train_loss_t, test_loss_t, confusion_matrix = cnn_trn.fit(model, loader, optimizer, loss_fn, nb_epochs)

cm_fig = cnn_trn.plot_confusion_matrix(confusion_matrix, classes=[0, 1, 2])
cm_fig.savefig("cm.png")

loss_fig = cnn_trn.plot_losses(train_loss_t, test_loss_t)
loss_fig.savefig("loss.png")

#cnn_trn.save_model_and_losses(train_loss_t, test_loss_t, model, model_name)


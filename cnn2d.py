from preprocessing import cnn2d_preprocessing as cnn_pre
from models import cnn2d_models as cnn_mod
from training import cnn2d_training as cnn_trn 

import git
import torch
import os


# import data
repo_not_cloned = not(os.path.isdir('RUL-Net'))
if repo_not_cloned:
    git.Git().clone("https://github.com/LahiruJayasinghe/RUL-Net.git")


loader = cnn_pre.Loader(use_cuda=True, dataset_number=3, batch_size=32)

model = Net(num_channels=loader.num_channels)

if loader.cuda:
    model = model.cuda()

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = nn.NLLLoss()

nb_epochs = 10

model_name = "CNN2d"

model, train_loss_t, test_loss_t = cnn_trn.fit(model, loader, optimizer, loss_fn, nb_epochs)

cnn_trn.plot_losses(train_loss_t, test_loss_t)

cnn_trn.save_model_and_losses(train_loss_t, test_loss_t, model, model_name)


# import modules

from preprocessing import wtte_rnn_preprocessing as wrp
from models import wtte_rnn_models as wrm
from training import wtte_rnn_training as wrt

import git
import torch
import os


# import data

repo_not_cloned = not(os.path.isdir('RUL-Net'))
if repo_not_cloned:
    git.Git().clone("https://github.com/LahiruJayasinghe/RUL-Net.git")


# main pipeline for GRUnet

dataset_train, dataset_test, features_col_name, target_col_name = wrp.get_CMAPSSData(1)
X_train, y_train, X_test, y_test = wrp.convert_train_and_test_to_appropriate_format(dataset_train, dataset_test,
                                                                                    features_col_name, target_col_name)
model = wrm.GRUnet(dim_input=len(features_col_name), dim_recurrent=2, dim_hidden=20)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
loss_fn = wrm.weibull_loss()
nb_epochs = 50
model_name = 'GRU_r2_h20'
model, train_loss_t, test_loss_t = wrt.fit(model, X_train, y_train, X_test, y_test, optimizer, loss_fn, nb_epochs)
wrt.plot_losses(train_loss_t, test_loss_t)
wrt.save_model_and_losses(train_loss_t, test_loss_t, model, model_name)


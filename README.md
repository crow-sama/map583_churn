This is a deep learning project aiming at predicting the arrival instants of a point process by prediction the Time-To-Event at each new instant.

## Project members
Amine Chaker

Ahmed Alouane

Emile Lucas

## Datasets
RUL engine dataset with multiple time series features for each engine. The test data contains the number of cycles until engine failure. The goal is to predict the number of cycles until failure.

a) For WTTE-RNN model, the data was taken from https://github.com/LahiruJayasinghe/RUL-Net
b) For CNN1d and CNN2d, the data used is from 
https://github.com/cerlymarco/MEDIUM_NoteBook/
Namely, three files (PM_test.txt, PM_train.txt, PM_truth.txt) should be downloaded from the repository and put in the data/ folder.

## Project structure  
1) models  # files defining the NN architectures
    a) wtte_rnn_models.py  # RNN
    b) cnn1d_models.py
    c) cnn2d_models.py
2) training  # training toolbox
    a) wtte_rnn_training.py  # for RNN  
    b) cnn1d_training.py
    c) cnn2d_training.py
3) preprocessing   # read data and convert it to suitable format according to the NN used  
    a) wtte_rnn_preprocessing.py   # for RNN  
    b) cnn1d_preprocessing.py
    c) cnn2d_preprocessing.py
4) wtte_rnn.py   # pipeline for importing data, running the RNN model: training the NN and saving it  
5) cnn1d.py # main file for testing
5) cnn2d.py # main file for testing the model (remove .npy cache files after modifying the loader parameters)

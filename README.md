This is a deep learning project aiming at predicting the arrival instants of a point process by prediction the Time-To-Event at each new instant.

## Project members
Amine Chaker

## Datasets

## Project structure  
1) models  # files defining the NN architectures  
    a) wtte_rnn_models.py  # RNN  
2) training  # training toolbox  
    a) wtte_rnn_training.py  # for RNN  
3) preprocessing   # read data and convert it to suitable format according to the NN used  
    a) wtte_rnn_preprocessing.py   # for RNN  
4) wtte_rnn.py   # pipeline for importing data, running the RNN model: training the NN and saving it  


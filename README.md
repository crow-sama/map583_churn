This is a deep learning project aiming at predicting the arrival instants of a point process by prediction the Time-To-Event at each new instant.

## Project members
Amine Chaker

## Datasets

## Project structure
├── models  # files defining the NN architectures  
|   └── wtte_rnn_models.py  # RNN  
├── training  # training toolbox  
|   └── wtte_rnn_training.py  # for RNN  
├── preprocessing   # read data and convert it to suitable format according to the NN used  
|   └── wtte_rnn_preprocessing.py   # for RNN  
├── wtte_rnn.py   # pipeline for importing data, running the RNN model: training the NN and saving it  

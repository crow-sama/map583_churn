import torch.nn as nn
import torch

class GRUnet(nn.Module):
    def __init__(self, dim_input, dim_recurrent, dim_hidden, dim_output=2):
        super(GRUnet, self).__init__()
        self.gru = nn.GRU(input_size = dim_input,
                          hidden_size = dim_hidden, num_layers = dim_recurrent)
        self.hidden_to_two = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        gru_output, _ = self.gru(x)
        gru_output = gru_output.squeeze(1)
        two_output = self.hidden_to_two(gru_output)
        return torch.exp(two_output)


class LSTMnet(nn.Module):
    def __init__(self, dim_input, dim_recurrent=1, dim_hidden=100, dim_output=2):
        super(LSTMnet, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input,
                            hidden_size=dim_hidden, num_layers=dim_recurrent)
        self.hidden_to_two = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        two_to_parameters = nn.Softplus()
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output.squeeze(1)
        two_output = self.hidden_to_two(lstm_output)
        return two_to_parameters(two_output)


# define loss class (weibull)

class weibull_loss(nn.Module):
    def __init__(self):
        super(weibull_loss, self).__init__()
        self.epsilon = 1e-6

    def forward(self, output, y):
        ya = (y + self.epsilon) / (output[:, 0])
        beta = output[:, 1]
        likelihoods = torch.log(beta) + beta * torch.log(ya) - ya ** beta
        return -likelihoods.mean()
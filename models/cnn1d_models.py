import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_channels):
        
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(num_channels, 64, 3)

        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x)

        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

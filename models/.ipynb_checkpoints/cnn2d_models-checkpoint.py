import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, num_channels):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)

        #self.conv3 = nn.Conv2d(32, 64, 3)
        #self.conv4 = nn.Conv2d(64, 64, 3)

        #self.fc1 = nn.Linear(5184, 256)
        self.fc1 = nn.Linear(16928, 256)
        self.fc2 = nn.Linear(256, 3)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.dropout2d(F.max_pool2d(x, kernel_size=2), p=0.5)

        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.dropout2d(F.max_pool2d(x, kernel_size=2), p=0.5)
        #x = x.view(-1, 5184)
        x = x.view(-1, 16928)
        
        x = F.relu(self.fc1(x))
        
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

import torch
import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self,input_size):
        super(RegressionMLP,self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(25, 1),
        )

    def forward(self,x):
        x = self.net(x)
        return x


class ClassficationMLP(nn.Module):
    def __init__(self,input_size):
        super(ClassficationMLP,self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50,25),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.net(x)
        return x
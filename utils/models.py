import torch.nn as nn
import torch


class RegressionMLP(nn.Module):
    def __init__(self, input_size):
        super(RegressionMLP, self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ClassficationMLP(nn.Module):
    def __init__(self, input_size):
        super(ClassficationMLP, self).__init__()
        self.input_size = input_size
        self.net = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def get_params(self):
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self):
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

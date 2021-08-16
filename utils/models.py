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
    def __init__(self, input_size, dropout):
        super(ClassficationMLP, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(50, 25),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(25, 2)
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


class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.input_size = input_size
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=(1,), stride=(1,)),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,), stride=(1,)),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=(3,), stride=(3,)),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9*32, 32),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        # x = x.view(x.shape[0], -1)
        x = self.fc(x)
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

import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self,input_size,hid_size):
        super(SimpleMLP,self).__init__()
        self.input_size = input_size
        self.hid_size = hid_size
        self.net = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hid_size, 1),
        )

    def forward(self,x):
        x = self.net(x)
        return x

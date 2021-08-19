import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.buffer import Buffer


def project(gxy, ger):
    cor = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - cor * ger


class AGEM:
    def __init__(self, config, device, model, loss, optimizer):
        self.config = config
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.buffer = Buffer(self.config['buffer_size'], self.device)
        self.grad_dims = []
        for p in self.model.parameters():
            self.grad_dims.append(p.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset, index):
        # Add data to the buffer
        num_samples = self.config['buffer_size'] // len(dataset)

        loader = DataLoader(dataset[index], batch_size=num_samples, shuffle=False)
        x, y = next(iter(loader))
        self.buffer.add_data(examples=x.to(self.device), labels=y.to(self.device))

import numpy as np
import torch
from utils.buffer import Buffer


class AGEM_R:
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

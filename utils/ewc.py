import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Ewc:
    def __init__(self, model, loss, config, optimizer, device):
        self.logsoft = nn.LogSoftmax(dim=1)
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.config = config
        self.loss = loss
        self.checkpoint = None
        self.fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.model.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        train_loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        fish = torch.zeros_like(self.model.get_params())

        for j, (x, y) in enumerate(train_loader):
            inputs = x.to(self.device)
            labels = y.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.optimizer.zero_grad()
                output = self.model(ex)
                loss = - F.binary_cross_entropy(output, lab.unsqueeze(0), reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss.backward()
                fish += exp_cond_prob * self.model.get_grads() ** 2

        fish /= (len(train_loader) * self.config['batch_size'])

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.config['gamma']
            self.fish += fish

        self.checkpoint = self.model.get_params().data.clone()

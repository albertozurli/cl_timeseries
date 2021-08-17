import torch


class SI:
    def __init__(self, model, loss, config, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.config = config
        self.loss = loss
        self.checkpoint = self.model.get_params().data.clone().to(self.device)
        self.big_omega = None
        self.small_omega = 0

    def penalty(self):
        if self.big_omega is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.big_omega*((self.model.get_params().data - self.checkpoint)**2)).sum()
            return penalty

    def end_task(self):
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.model.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.model.get_params().data - self.checkpoint)**2 + self.config['xi'])

        self.checkpoint = self.model.get_params().data.clone().to(self.device)
        self.small_omega = 0

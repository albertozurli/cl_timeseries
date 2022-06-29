import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import backward_transfer, forgetting, forward_transfer
from utils.evaluation import evaluate_past, test_epoch, evaluate_next
from utils.utils import binary_accuracy
import pandas as pd


def train_si(model, loss, device, optimizer, train_set, test_set, suffix, config):
    si = SI(model, loss, config, optimizer, device)
    accuracy = []

    if config['evaluate']:
        text_file = open("si_" + suffix + ".txt", "a")
        text_file.write("SI LEARNING \n")
        test_list = [[] for _ in range(len(train_set))]

    # Eval without training
    _, _, random_mean_accuracy, _ = evaluate_past(si.model, len(test_set) - 1, test_set, si.loss, device)

    # Train
    for index, data_set in enumerate(train_set):
        si.model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            si.model.train()

            epoch_loss = []
            epoch_acc = []
            for j, (x, y) in enumerate(train_loader):
                si.optimizer.zero_grad()

                x = x.to(device)
                output = si.model(x)
                y = y.to(device)

                penalty = si.penalty()
                s_loss = si.loss(output, y.squeeze(1)) + config['c'] * penalty

                if config['cnn']:
                    l1_reg = 0
                    for param in si.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    s_loss += config['l1_lambda'] * l1_reg

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, y.squeeze(1))
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                nn.utils.clip_grad.clip_grad_value_(si.model.parameters(), 1)
                si.optimizer.step()
                si.small_omega += config['lr'] * si.model.get_grads().data ** 2

            if (epoch % 100 == 0) or (epoch == (config['epochs'] - 1)):
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(si.model, test_loader, si.loss, device)
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(si.model, test_loader, si.loss, device)
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

        si.end_task()

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(si.model, index, test_set, si.loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(si.model, index, test_set, si.loss, device))

    # Compute transfer metrics
    backward = backward_transfer(accuracy)
    forward = forward_transfer(accuracy, random_mean_accuracy)
    forget = forgetting(accuracy)
    print(f'Backward transfer: {backward}')
    print(f'Forward transfer: {forward}')
    print(f'Forgetting: {forget}')

    if config['evaluate']:
        text_file.write(f"Backward: {backward}\n")
        text_file.write(f"Forward: {forward}\n")
        text_file.write(f"Forgetting: {forget}\n")
        text_file.close()

        df = pd.DataFrame(test_list)
        df.to_csv(f'si_{suffix}.csv')


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
            penalty = (self.big_omega * ((self.model.get_params().data - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self):
        if self.big_omega is None:
            self.big_omega = torch.zeros_like(self.model.get_params()).to(self.device)

        self.big_omega += self.small_omega / ((self.model.get_params().data - self.checkpoint) ** 2 + self.config['xi'])

        self.checkpoint = self.model.get_params().data.clone().to(self.device)
        self.small_omega = 0

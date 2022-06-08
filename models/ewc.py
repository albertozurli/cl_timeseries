import statistics
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metrics import backward_transfer, forgetting, forward_transfer
from utils.evaluation import evaluate_past, test_epoch, evaluate_next
from utils.utils import binary_accuracy
import wandb
import pandas as pd
import torch.nn.functional as F


def train_ewc(model, loss, device, optimizer, train_set, test_set, suffix, config):
    wandb.init(project="LOD2022", entity="albertozurli", reinit=True)
    ewc = EWC(model, loss, config, optimizer, device)
    accuracy = []

    # N SummaryWriter for N domains
    if config['evaluate']:
        text_file = open("ewc_" + suffix + ".txt", "a")
        text_file.write("EWC LEARNING \n")
        test_list = [[] for _ in range(len(train_set))]

    # Eval without training
    _, _, random_mean_accuracy, _ = evaluate_past(ewc.model, len(test_set) - 1, test_set, ewc.loss, device)

    # Train
    for index, data_set in enumerate(train_set):
        ewc.model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            ewc.model.train()

            epoch_loss = []
            epoch_acc = []
            for j, (x, y) in enumerate(train_loader):
                ewc.optimizer.zero_grad()

                x = x.to(device)
                output = ewc.model(x)
                y = y.to(device)

                penalty = ewc.penalty()
                s_loss = ewc.loss(output, y.squeeze(1)) + (config['e_lambda'] * penalty)

                if config['cnn']:
                    l1_reg = 0
                    for param in ewc.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    s_loss += config['l1_lambda'] * l1_reg

                assert not torch.isnan(s_loss)

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, y.squeeze(1))
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                ewc.optimizer.step()

            wandb.log({"Train/loss": statistics.mean(epoch_loss),
                       "Train/accuracy": statistics.mean(epoch_acc)})

            if (epoch % 100 == 0) or (epoch == (config['epochs'] - 1)):
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(ewc.model, test_loader, ewc.loss, device)
                    wandb.log({f"Test/domain{past}_acc":statistics.mean(tmp)})
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(ewc.model, test_loader, ewc.loss, device)
                wandb.log({f"Test/domain{index}_acc": statistics.mean(tmp),
                           "Test/domain_loss": statistics.mean(loss_task)})
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

                avg = sum(tmp_list) / len(tmp_list)
                wandb.log({"Test/mean_acc": avg})
        ewc.end_task(data_set)

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(ewc.model, index, test_set, ewc.loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(ewc.model, index, test_set, ewc.loss, device))

        torch.save(ewc.model.state_dict(), f'checkpoints/ewc/model_d{index}.pt')

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
        df.to_csv(f'ewc_{suffix}.csv')


class EWC:
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
                output = self.model(ex.unsqueeze(0))
                loss = - F.cross_entropy(output, lab, reduction='none')
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

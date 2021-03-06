import statistics
import torch
import quadprog
from utils.buffer import Buffer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import backward_transfer, forgetting, forward_transfer
from utils.evaluation import evaluate_past, test_epoch, evaluate_next
from utils.utils import binary_accuracy, unique
import pandas as pd
import numpy as np


def train_gem(model, loss, device, optimizer, train_set, test_set, suffix, config):
    gem = GEM(config, device, model, loss, optimizer)
    accuracy = []

    if config['evaluate']:
        text_file = open("gem_" + suffix + ".txt", "a")
        text_file.write("GEM LEARNING \n")
        test_list = [[] for _ in range(len(train_set))]

    # Eval without training
    _, _, random_mean_accuracy, _ = evaluate_past(gem.model, len(test_set) - 1, test_set, gem.loss, device)

    # Train
    for index, data_set in enumerate(train_set):
        gem.model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            gem.model.train()

            epoch_loss = []
            epoch_acc = []
            for j, (x, y) in enumerate(train_loader):

                if not gem.buffer.is_empty():
                    buf_inputs, buf_labels, buf_task_labels = gem.buffer.get_data(config['buffer_size'],
                                                                                  task_labels=True)

                    # Gradient buffer
                    for tt in unique(buf_task_labels):
                        gem.optimizer.zero_grad()
                        cur_task_inputs = [buf_inputs[i] for i in range(len(buf_task_labels)) if
                                           buf_task_labels[i] == tt]
                        cur_task_labels = [buf_labels[i] for i in range(len(buf_task_labels)) if
                                           buf_task_labels[i] == tt]
                        cur_task_outputs = gem.model(torch.stack(cur_task_inputs))
                        buffer_loss = gem.loss(cur_task_outputs, torch.stack(cur_task_labels).squeeze(1))
                        buffer_loss.backward()
                        store_gradient(gem.model.parameters(), gem.grads_cs[tt], gem.grad_dims)

                # Gradient current task
                gem.optimizer.zero_grad()
                x = x.to(device)
                output = gem.model(x)
                y = y.to(device)
                s_loss = gem.loss(output, y.squeeze(1))

                if config['cnn']:
                    l1_reg = 0
                    for param in gem.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    s_loss += config['l1_lambda'] * l1_reg

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, y.squeeze(1))
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()

                # Check if gradient violates buffer constraints
                if not gem.buffer.is_empty():
                    store_gradient(gem.model.parameters(), gem.grads_da, gem.grad_dims)

                    dot_prod = torch.mm(gem.grads_da.unsqueeze(0), torch.stack(gem.grads_cs).T)
                    if (dot_prod < 0).sum() != 0:
                        project2cone2(gem.grads_da.unsqueeze(1), torch.stack(gem.grads_cs).T,
                                      margin=config['gem_gamma'])
                        # Copy gradient
                        overwrite_gradient(gem.model.parameters(), gem.grads_da, gem.grad_dims)

                gem.optimizer.step()

            if (epoch % 100 == 0) or (epoch == (config['epochs'] - 1)):
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(gem.model, test_loader, gem.loss, device)
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(gem.model, test_loader, gem.loss, device)
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

        gem.end_task(train_set)

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(gem.model, index, test_set, gem.loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(gem.model, index, test_set, gem.loss, device))

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
        df.to_csv(f"gem_{suffix}.csv")


def store_gradient(parameters, gradient, gradient_dims):
    gradient.fill_(0.0)
    count = 0
    for p in parameters:
        if p.grad is not None:
            begin = 0 if count == 0 else sum(gradient_dims[:count])
            end = np.sum(gradient_dims[:count + 1])
            gradient[begin:end].copy_(p.grad.data.view(-1))
        count += 1


def overwrite_gradient(parameters, new, gradient_dims):
    count = 0
    for p in parameters:
        if p.grad is not None:
            begin = 0 if count == 0 else sum(gradient_dims[:count])
            end = np.sum(gradient_dims[:count + 1])
            new_grad = new[begin:end].contiguous().view(p.grad.data.size())
            p.grad.data.copy_(new_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


class GEM:
    def __init__(self, config, device, model, loss, optimizer):
        self.current_task = 0
        self.config = config
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.buffer = Buffer(self.config['buffer_size'], self.device)

        self.grad_dims = []
        for pp in self.model.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def end_task(self, dataset):
        self.current_task += 1
        self.grads_cs.append(torch.zeros(np.sum(self.grad_dims)).to(self.device))

        # Add data to the buffer
        num_samples = self.config['buffer_size'] // len(dataset)

        loader = DataLoader(dataset[(self.current_task - 1)], batch_size=num_samples, shuffle=False)
        x, y = next(iter(loader))
        self.buffer.add_data(examples=x.to(self.device),
                             task=torch.ones(1, dtype=torch.long).to(self.device) * (self.current_task - 1),
                             labels=y.to(self.device))

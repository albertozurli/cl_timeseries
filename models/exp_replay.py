import statistics
import torch
from utils.buffer import Buffer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import backward_transfer, forgetting, forward_transfer
from utils.evaluation import evaluate_past, test_epoch, evaluate_next
from utils.utils import binary_accuracy
import pandas as pd


def train_er(train_set, test_set, model, loss, optimizer, device, config, suffix):

    er = ER(config,device,model,loss,optimizer)
    accuracy = []

    if config['evaluate']:
        text_file = open("er_" + suffix + ".txt", "a")
        text_file.write("\nCONTINUAL LEARNING W\\ ER \n")
        test_list = [[] for _ in range(len(train_set))]

    # Eval without training
    _, _, random_mean_accuracy, _ = evaluate_past(er.model, len(test_set) - 1, test_set, loss, device)

    for index, data_set in enumerate(train_set):
        er.model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            er.model.train()
            epoch_loss = []
            epoch_acc = []
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = x.to(device)
                labels = y.to(device)
                if not er.buffer.is_empty():
                    # Strategy 50/50
                    # From batch of 64 (dataloader) to 64 + 64 (dataloader + replay)
                    buf_input, buf_label, _ = er.buffer.get_data(config['batch_size'])
                    inputs = torch.cat((inputs, torch.stack(buf_input)))
                    labels = torch.cat((labels, torch.stack(buf_label)))

                output = er.model(inputs)
                s_loss = loss(output, labels.squeeze(1))

                if config['cnn']:
                    l1_reg = 0
                    for param in er.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    s_loss += config['l1_lambda'] * l1_reg

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, labels.squeeze(1))

                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

                if epoch == 0:
                    er.buffer.add_data(examples=x.to(device), labels=y.to(device))

            if (epoch % 100 == 0) or (epoch == (config['epochs'] - 1)):
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(er.model, test_loader, loss, device)
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(er.model, test_loader, loss, device)
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(er.model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
                text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                                f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(er.model, index, test_set, loss, device))

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
        df.to_csv(f'er_{suffix}.csv')


class ER:
    def __init__(self, config, device, model, loss, optimizer):
        self.config = config
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.buffer = Buffer(self.config['buffer_size'], self.device)

import statistics
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.evaluation import test_epoch, evaluate_past
from utils.utils import binary_accuracy


def train_online(train_set, test_set, model, loss, optimizer, device, config, suffix):

    if config['evaluate']:
        text_file = open("online_" + suffix + ".txt", "a")
        text_file.write("ONLINE LEARNING \n")
        test_list = [[] for _ in range(len(train_set))]

    # Train
    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for i in tqdm(range(config['epochs'])):
            model.train()

            epoch_loss = []
            epoch_acc = []

            for j, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)
                output = model(x)
                s_loss = loss(output, y.squeeze(1))

                if config['cnn']:
                    l1_reg = 0
                    for param in model.parameters():
                        l1_reg += torch.norm(param, 1)
                    s_loss += config['l1_lambda'] * l1_reg

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred, y.squeeze(1))

                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

            if (i % 100 == 0) or (i == (config['epochs'] - 1)):
                print(f'\nEpoch {i:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Test each epoch
            if config['evaluate']:
                tmp_list = []
                # Past tasks
                for past in range(index):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp, _ = test_epoch(model, test_loader, loss, device)
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(model, test_loader, loss, device)
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

    if config['evaluate']:
        text_file.close()
        df = pd.DataFrame(test_list)
        df.to_csv(f'online_{suffix}.csv')

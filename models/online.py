import statistics

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from utils.evaluation import test_epoch, evaluate_past
from utils.utils import binary_accuracy


def train_online(train_set: object, test_set: object, model: object, loss: object, optimizer: object, device: object, config: object, suffix: object) -> object:
    """
    :param train_set: Train set
    :param test_set: Test set
    :param model: PyTorch model
    :param loss: loss function
    :param optimizer: optimizer
    :param device: device cuda/cpu
    :param config: configuration
    :param suffix: Suffix for the filename and Summary Writer
    """
    wandb.init(project="LOD2022", entity="albertozurli", reinit=True)
    # train_writer = SummaryWriter('./runs/online/train/' + suffix)

    # N SummaryWriter for N domains
    if config['evaluate']:
        text_file = open("online_" + suffix + ".txt", "a")
        text_file.write("ONLINE LEARNING \n")
        # test_writer = SummaryWriter('./runs/online/test/' + suffix)
        # writer_list = []
        test_list = [[] for _ in range(len(train_set))]
        # for i in range(len(train_set)):
        #     writer_list.append(SummaryWriter(f'./runs/online/test/{suffix}/d_{i}'))

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

            # train_writer.add_scalar('Train/total_loss',
            #                         statistics.mean(epoch_loss), i + (config['epochs'] * index))
            # train_writer.add_scalar('Train/accuracy',
            #                         statistics.mean(epoch_acc), i + (config['epochs'] * index))
            wandb.log({"Train/loss":statistics.mean(epoch_loss),
                       "Train/accuracy":statistics.mean(epoch_acc)})

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
                    wandb.log({f"Test/domain{past}_acc":statistics.mean(tmp)})
                    # writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                    #                              i + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))
                    for t in tmp:
                        tmp_list.append(t)
                # Current task
                test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)
                tmp, loss_task = test_epoch(model, test_loader, loss, device)
                wandb.log({f"Test/domain{index}_acc":statistics.mean(tmp),
                           "Test/domain_loss":statistics.mean(loss_task)})
                # writer_list[index].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                #                               i + (config['epochs'] * index))
                # test_writer.add_scalar('Test/domain_loss', statistics.mean(loss_task),
                #                               i + (config['epochs'] * index))
                test_list[index].append(statistics.mean(tmp))
                for t in tmp:
                    tmp_list.append(t)

                avg = (sum(tmp_list) / len(tmp_list))
                wandb.log({"Test/mean_acc":avg})
                # test_writer.add_scalar('Test/mean_accuracy', avg, i + (config['epochs'] * index))

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.5f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.5f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.5f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        torch.save(model.state_dict(), f'checkpoints/online/model_d{index}.pt')

    if config['evaluate']:
        text_file.close()

        df = pd.DataFrame(test_list)
        df.to_csv(f'online_{suffix}.csv')

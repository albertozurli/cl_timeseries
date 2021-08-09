import torch
import statistics

import pandas as pd
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.buffer import Buffer
from utils.ewc import Ewc
from utils.si import Si
from utils.utils import binary_accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import backward_transfer, forward_transfer, forgetting


def test_epoch(model, test_loader, device):
    """
    :param model: PyTorch model
    :param test_loader: DataLoader
    :param device: device (cuda/cpu)
    """
    model.eval()
    test_acc = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, pred = torch.max(output.data, 1)
            acc = binary_accuracy(pred.float(), y)
            test_acc.append(acc.item())

    return test_acc


def test(model, loss, test_loader, device):
    """
    :param model: PyTorch model
    :param loss: loss function
    :param test_loader: DataLoader
    :param device: device (cuda/cpu)
    """
    model.eval()
    test_loss = []
    test_acc = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, pred = torch.max(output.data, 1)
            s_loss = loss(output, y.long())
            acc = binary_accuracy(pred.float(), y)
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Error: {statistics.mean(test_loss):.2f} | Acc: {statistics.mean(test_acc):.2f}%")
    return test_acc, test_loss


def evaluate_next(model, domain, test_set, loss, device):
    print("---Eval next domain---")
    test_loader = DataLoader(test_set[domain + 1], batch_size=1, shuffle=False)
    accuracy, _ = test(model, loss, test_loader, device)
    return statistics.mean(accuracy)


def evaluate_past(model, domain, test_set, loss, device):
    accs = []
    errors = []
    mean_accs = []
    mean_errors = []
    print("---Eval past domains---")
    for past in range(domain + 1):
        print(f"Domain {past} | ", end="")
        test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
        accuracy, error = test(model, loss, test_loader, device)
        accs.append(accuracy)
        errors.append(error)
        mean_accs.append(statistics.mean(accuracy))
        mean_errors.append(statistics.mean(error))
    flat_accs = [item for sublist in accs for item in sublist]
    flat_errors = [item for sublist in errors for item in sublist]
    return flat_accs, flat_errors, mean_accs, mean_errors


def train_dark_er(train_set, test_set, model, loss, optimizer, device, config, suffix):
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

    train_writer = SummaryWriter('./runs/dark_exp_replay/train/' + suffix)

    buffer = Buffer(config['buffer_size'], device)
    accuracy = []

    # Save results in a .txt file (write or append)
    if config['evaluate']:
        text_file = open("dark_er_" + suffix + ".txt", "a")
        text_file.write("\nCONTINUAL LEARNING W\\ DER \n")
        text_file.write("---Eval pre-training--- \n")

    # Eval without training
    print("-----EVAL PRE-TRAINING-----")
    random_accuracy, random_error, random_mean_accuracy, random_mean_error = evaluate_past(model, len(test_set) - 1,
                                                                                           test_set, loss, device)
    print(f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}%")

    if config['evaluate']:
        for i, a in enumerate(random_mean_accuracy):
            text_file.write(f"Domain {i} | Error: {random_mean_error[i]:.2f} | Acc: {a:.2f}%\n")
        text_file.write(
            f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}% \n")

    # N SummaryWriter for N domains
    if config['evaluate']:
        test_writer = SummaryWriter('./runs/dark_exp_replay/test/' + suffix)
        writer_list = []
        test_list = [[] for _ in range(len(train_set) + 1)]
        mean_acc_list = []
        for i in range(len(train_set)):
            writer_list.append(SummaryWriter(f'./runs/dark_exp_replay/test/{suffix}/d_{i}'))

    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            model.train()
            epoch_loss = []
            epoch_acc = []
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = x.to(device)
                labels = y.to(device)
                output = model(inputs)

                _, pred = torch.max(output.data, 1)
                s_loss = loss(output, labels.long())

                if not buffer.is_empty():
                    # Strategy 50/50
                    # From batch of 64 (dataloader) to 64 + 64 (dataloader + replay)
                    buf_input, buf_logit = buffer.get_data(config['batch_size'])
                    buf_output = model(torch.stack(buf_input))
                    add_loss = F.mse_loss(buf_output, torch.stack(buf_logit))
                    s_loss += config['alpha'] * add_loss.data

                acc = binary_accuracy(pred.float(), labels)

                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

                if epoch == 0:
                    buffer.add_data(examples=x.to(device), labels=output.to(device))

            train_writer.add_scalar('Train/loss', statistics.mean(epoch_loss),
                                    epoch + (config['epochs'] * index))
            train_writer.add_scalar('Train/accuracy', statistics.mean(epoch_acc),
                                    epoch + (config['epochs'] * index))

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Last epoch (only for stats)
            if epoch == 499:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}% \n')

            # Test each epoch
            if config['evaluate']:
                tmp_bucket = []
                for past in range(index + 1):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp = test_epoch(model, test_loader, device)
                    tmp_bucket.append(tmp)
                    writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                                                 epoch + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))

                tmp_bucket = [item for sublist in tmp_bucket for item in sublist]
                avg = sum(tmp_bucket) / len(tmp_bucket)
                mean_acc_list.append(avg)
                test_writer.add_scalar('Test/mean_accuracy', avg, epoch + (config['epochs'] * index))

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.2f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.2f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.2f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(model, index, test_set, loss, device))

        torch.save(model.state_dict(), f'checkpoints/der/model_d{index}.pt')

    # Check buffer distribution
    buffer.check_distribution()

    # Compute transfer metrics
    backward = backward_transfer(accuracy)
    forward = forward_transfer(accuracy, random_mean_accuracy)
    forget = forgetting(accuracy)
    print(f'\nBackward transfer: {backward}')
    print(f'Forward transfer: {forward}')
    print(f'Forgetting: {forget}')

    if config['evaluate']:
        text_file.write(f"Backward: {backward}\n")
        text_file.write(f"Forward: {forward}\n")
        text_file.write(f"Forgetting: {forget}\n")
        text_file.close()

        test_list[len(test_list) - 1].append(mean_acc_list)

        df = pd.DataFrame(test_list)
        df.to_csv(f'{suffix}_der.csv')


def train_er(train_set, test_set, model, loss, optimizer, device, config, suffix):
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

    train_writer = SummaryWriter('./runs/exp_replay/train/' + suffix)
    buffer = Buffer(config['buffer_size'], device)
    accuracy = []

    # Save results in a .txt file (write or append)
    if config['evaluate']:
        text_file = open("er_" + suffix + ".txt", "a")
        text_file.write("\nCONTINUAL LEARNING W\\ ER \n")
        text_file.write("---Eval pre-training--- \n")

    # Eval without training
    print("-----EVAL PRE-TRAINING-----")
    random_accuracy, random_error, random_mean_accuracy, random_mean_error = evaluate_past(model, len(test_set) - 1,
                                                                                           test_set, loss, device)
    print(f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}%")

    if config['evaluate']:
        for i, a in enumerate(random_mean_accuracy):
            text_file.write(f"Domain {i} | Error: {random_mean_error[i]:.2f} | Acc: {a:.2f}%\n")
        text_file.write(
            f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}% \n")

    # N SummaryWriter for N domains
    if config['evaluate']:
        test_writer = SummaryWriter('./runs/exp_replay/test/' + suffix)
        writer_list = []
        test_list = [[] for _ in range(len(train_set) + 1)]
        mean_acc_list = []
        for i in range(len(train_set)):
            writer_list.append(SummaryWriter(f'./runs/exp_replay/test/{suffix}/d_{i}'))

    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            model.train()
            epoch_loss = []
            epoch_acc = []
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = x.to(device)
                labels = y.to(device)
                if not buffer.is_empty():
                    # Strategy 50/50
                    # From batch of 64 (dataloader) to 64 + 64 (dataloader + replay)
                    buf_input, buf_label = buffer.get_data(config['batch_size'])
                    inputs = torch.cat((inputs, torch.stack(buf_input)))
                    labels = torch.cat((labels, torch.stack(buf_label)))

                output = model(inputs)
                _, pred = torch.max(output.data, 1)
                s_loss = loss(output, labels.long())
                acc = binary_accuracy(pred.float(), labels)

                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

                if epoch == 0:
                    buffer.add_data(examples=x.to(device), labels=y.to(device))

            train_writer.add_scalar('Train/loss', statistics.mean(epoch_loss),
                                    epoch + (config['epochs'] * index))
            train_writer.add_scalar('Train/accuracy', statistics.mean(epoch_acc),
                                    epoch + (config['epochs'] * index))

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Last epoch (only for stats)
            if epoch == 499:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}% \n')

            # Test each epoch
            if config['evaluate']:
                tmp_bucket = []
                for past in range(index + 1):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp = test_epoch(model, test_loader, device)
                    tmp_bucket.append(tmp)
                    writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                                                 epoch + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))

                tmp_bucket = [item for sublist in tmp_bucket for item in sublist]
                avg = sum(tmp_bucket) / len(tmp_bucket)
                mean_acc_list.append(avg)
                test_writer.add_scalar('Test/mean_accuracy', avg, epoch + (config['epochs'] * index))

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.2f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.2f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.2f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(model, index, test_set, loss, device))

        torch.save(model.state_dict(), f'checkpoints/er/model_d{index}.pt')

    # Check buffer distribution
    buffer.check_distribution()

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

        test_list[len(test_list) - 1].append(mean_acc_list)

        df = pd.DataFrame(test_list)
        df.to_csv(f'{suffix}_er.csv')


def train_online(train_set, test_set, model, loss, optimizer, device, config, suffix):
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

    train_writer = SummaryWriter('./runs/online/train/' + suffix)

    if config['evaluate']:
        text_file = open("online_" + suffix + ".txt", "a")
        text_file.write("ONLINE LEARNING \n")
        text_file.write("---Eval pre-training--- \n")

    # Eval without training
    print("-----EVAL PRE-TRAINING-----")
    random_accuracy, random_error, random_mean_accuracy, random_mean_error \
        = evaluate_past(model, len(test_set) - 1, test_set, loss, device)
    print(f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}%")
    if config['evaluate']:
        for i, a in enumerate(random_mean_accuracy):
            text_file.write(f"Domain {i} | Error: {random_mean_error[i]:.2f} | Acc: {a:.2f}%\n")
        text_file.write(f"Mean Error: {statistics.mean(random_error):.2f} |"
                        f" Mean Acc: {statistics.mean(random_accuracy):.2f}% \n")

    # N SummaryWriter for N domains
    if config['evaluate']:
        test_writer = SummaryWriter('./runs/online/test/' + suffix)
        writer_list = []
        test_list = [[] for _ in range(len(train_set) + 1)]
        mean_acc_list = []
        for i in range(len(train_set)):
            writer_list.append(SummaryWriter(f'./runs/online/test/{suffix}/d_{i}'))

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
                output = model(x)
                y = y.to(device)
                s_loss = loss(output, y.long())
                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred.float(), y)

                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

            train_writer.add_scalar('Train/loss',
                                    statistics.mean(epoch_loss), i + (config['epochs'] * index))
            train_writer.add_scalar('Train/accuracy',
                                    statistics.mean(epoch_acc), i + (config['epochs'] * index))

            if i % 100 == 0:
                print(f'\nEpoch {i:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

            # Last epoch (only for stats)
            if i == 499:
                print(f'\nEpoch {i:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}% \n')

            # Test each epoch
            if config['evaluate']:
                tmp_bucket = []
                for past in range(index + 1):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp = test_epoch(model, test_loader, device)
                    tmp_bucket.append(tmp)
                    writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                                                 i + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))

                tmp_bucket = [item for sublist in tmp_bucket for item in sublist]
                avg = sum(tmp_bucket) / len(tmp_bucket)
                mean_acc_list.append(avg)
                test_writer.add_scalar('Test/mean_accuracy', avg, i + (config['epochs'] * index))

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.2f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.2f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.2f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        torch.save(model.state_dict(), f'checkpoints/online/model_d{index}.pt')

    if config['evaluate']:
        text_file.close()

        test_list[len(test_list) - 1].append(mean_acc_list)

        df = pd.DataFrame(test_list)
        df.to_csv(f'{suffix}_online.csv')


def train_ewc(model, loss, device, optimizer, train_set, test_set, suffix, config):
    train_writer = SummaryWriter('./runs/ewc/train/' + suffix)
    ewc = Ewc(model, loss, config, optimizer, device)
    accuracy = []

    if config['evaluate']:
        text_file = open("ewc_" + suffix + ".txt", "a")
        text_file.write("EWC LEARNING \n")
        text_file.write("---Eval pre-training--- \n")

    # Eval without training
    print("-----EVAL PRE-TRAINING-----")

    random_accuracy, random_error, random_mean_accuracy, random_mean_error \
        = evaluate_past(model, len(test_set) - 1, test_set, loss, device)
    print(f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}%")
    if config['evaluate']:
        for i, a in enumerate(random_mean_accuracy):
            text_file.write(f"Domain {i} | Error: {random_mean_error[i]:.2f} | Acc: {a:.2f}%\n")
        text_file.write(f"Mean Error: {statistics.mean(random_error):.2f} |"
                        f" Mean Acc: {statistics.mean(random_accuracy):.2f}% \n")

    # N SummaryWriter for N domains
    if config['evaluate']:
        test_writer = SummaryWriter('./runs/ewc/test/' + suffix)
        writer_list = []
        test_list = [[] for _ in range(len(train_set) + 1)]
        mean_acc_list = []
        for i in range(len(train_set)):
            writer_list.append(SummaryWriter(f'./runs/ewc/test/{suffix}/d_{i}'))

    # Train
    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            model.train()

            epoch_loss = []
            epoch_acc = []
            for j, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                x = x.to(device)
                output = model(x)
                y = y.to(device)

                penalty = ewc.penalty()
                s_loss = loss(output, y.long()) + (config['e_lambda'] * penalty)
                assert not torch.isnan(s_loss)

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred.float(), y)
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

            train_writer.add_scalar('Train/loss', statistics.mean(epoch_loss),
                                    epoch + (config['epochs'] * index))
            train_writer.add_scalar('Train/accuracy', statistics.mean(epoch_acc),
                                    epoch + (config['epochs'] * index))

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

                # Last epoch (only for stats)
            if epoch == 499:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}% \n')

            # Test each epoch
            if config['evaluate']:
                tmp_bucket = []
                for past in range(index + 1):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp = test_epoch(model, test_loader, device)
                    tmp_bucket.append(tmp)
                    writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                                                 epoch + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))

                tmp_bucket = [item for sublist in tmp_bucket for item in sublist]
                avg = sum(tmp_bucket) / len(tmp_bucket)
                mean_acc_list.append(avg)
                test_writer.add_scalar('Test/mean_accuracy', avg, epoch + (config['epochs'] * index))

        ewc.end_task(data_set)

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.2f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.2f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.2f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(model, index, test_set, loss, device))

        torch.save(model.state_dict(), f'checkpoints/ewc/model_d{index}.pt')

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

        test_list[len(test_list) - 1].append(mean_acc_list)

        df = pd.DataFrame(test_list)
        df.to_csv(f'{suffix}_ewc.csv')


def train_si(model, loss, device, optimizer, train_set, test_set, suffix, config):
    train_writer = SummaryWriter('./runs/si/train/' + suffix)
    si = Si(model, loss, config, optimizer, device)
    accuracy = []

    if config['evaluate']:
        text_file = open("si_" + suffix + ".txt", "a")
        text_file.write("SI LEARNING \n")
        text_file.write("---Eval pre-training--- \n")

    # Eval without training
    print("-----EVAL PRE-TRAINING-----")

    random_accuracy, random_error, random_mean_accuracy, random_mean_error \
        = evaluate_past(model, len(test_set) - 1, test_set, loss, device)
    print(f"Mean Error: {statistics.mean(random_error):.2f} | Mean Acc: {statistics.mean(random_accuracy):.2f}%")
    if config['evaluate']:
        for i, a in enumerate(random_mean_accuracy):
            text_file.write(f"Domain {i} | Error: {random_mean_error[i]:.2f} | Acc: {a:.2f}%\n")
        text_file.write(f"Mean Error: {statistics.mean(random_error):.2f} |"
                        f" Mean Acc: {statistics.mean(random_accuracy):.2f}% \n")

    # N SummaryWriter for N domains
    if config['evaluate']:
        test_writer = SummaryWriter('./runs/si/test/' + suffix)
        writer_list = []
        test_list = [[] for _ in range(len(train_set) + 1)]
        mean_acc_list = []
        for i in range(len(train_set)):
            writer_list.append(SummaryWriter(f'./runs/si/test/{suffix}/d_{i}'))

    # Train
    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        train_loader = DataLoader(data_set, batch_size=config["batch_size"], shuffle=False)

        for epoch in tqdm(range(config['epochs'])):
            model.train()

            epoch_loss = []
            epoch_acc = []
            for j, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                x = x.to(device)
                output = model(x)
                y = y.to(device)

                penalty = si.penalty()
                s_loss = loss(output, y.long()) + config['c'] * penalty

                _, pred = torch.max(output.data, 1)
                acc = binary_accuracy(pred.float(), y)
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                nn.utils.clip_grad.clip_grad_value_(model.parameters(), 1)
                optimizer.step()
                si.small_omega += config['lr'] * model.get_grads().data ** 2

            train_writer.add_scalar('Train/loss', statistics.mean(epoch_loss),
                                    epoch + (config['epochs'] * index))
            train_writer.add_scalar('Train/accuracy', statistics.mean(epoch_acc),
                                    epoch + (config['epochs'] * index))

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}%')

                # Last epoch (only for stats)
            if epoch == 499:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.2f} '
                      f'| Acc: {statistics.mean(epoch_acc):.2f}% \n')

            # Test each epoch
            if config['evaluate']:
                tmp_bucket = []
                for past in range(index + 1):
                    test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
                    tmp = test_epoch(model, test_loader, device)
                    tmp_bucket.append(tmp)
                    writer_list[past].add_scalar('Test/domain_accuracy', statistics.mean(tmp),
                                                 epoch + (config['epochs'] * index))
                    test_list[past].append(statistics.mean(tmp))

                tmp_bucket = [item for sublist in tmp_bucket for item in sublist]
                avg = sum(tmp_bucket) / len(tmp_bucket)
                mean_acc_list.append(avg)
                test_writer.add_scalar('Test/mean_accuracy', avg, epoch + (config['epochs'] * index))

        si.end_task()

        # Test at the end of domain
        evaluation, error, mean_evaluation, mean_error = evaluate_past(model, index, test_set, loss, device)
        print(f"Mean Error: {statistics.mean(error):.2f} | Mean Acc: {statistics.mean(evaluation):.2f}%")
        accuracy.append(mean_evaluation)
        if config['evaluate']:
            text_file.write(f"---Evaluation after domain {index}--- \n")
            for i, a in enumerate(mean_evaluation):
                text_file.write(f"Domain {i} | Error: {mean_error[i]:.2f} | Acc: {a:.2f}%\n")
            text_file.write(f"Mean Error: {statistics.mean(error):.2f} | "
                            f"Mean Acc: {statistics.mean(evaluation):.2f}% \n")

        if index != len(train_set) - 1:
            accuracy[index].append(evaluate_next(model, index, test_set, loss, device))

        torch.save(model.state_dict(), f'checkpoints/si/model_d{index}.pt')

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

        test_list[len(test_list) - 1].append(mean_acc_list)

        df = pd.DataFrame(test_list)
        df.to_csv(f'{suffix}_si.csv')

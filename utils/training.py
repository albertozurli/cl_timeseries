import torch
import statistics
import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.buffer import Buffer
from utils.utils import binary_accuracy
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import backward_transfer, forward_transfer, forgetting


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
    predicted = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            s_loss = loss(pred.squeeze(1), y)
            acc = binary_accuracy(pred.squeeze(1), y)

            predicted.append(pred.cpu().numpy().item())
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Test error: {statistics.mean(test_loss):.5f} | Test accuracy: {statistics.mean(test_acc):.5f}")
    return statistics.mean(test_acc)


def evaluate_next(model, domain, test_set, loss, device):
    test_loader = DataLoader(test_set[domain + 1], batch_size=1, shuffle=False)
    accuracy = test(model, loss, test_loader, device)
    return accuracy


def evaluate_past(model, domain, test_set, loss, device):
    accs = []
    for past in range(domain + 1):
        print(f"Testing model on domain {past}")
        test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
        accuracy = test(model, loss, test_loader, device)
        accs.append(accuracy)
    return accs


def train_cl(train_set, test_set, model, loss, optimizer, device, config):
    """
    :param train_set: Train set
    :param test_set: Test set
    :param model: PyTorch model
    :param loss: loss function
    :param optimizer: optimizer
    :param device: device cuda/cpu
    :param config: configuration
    """

    global_writer = SummaryWriter('./runs/continual/train/global/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))
    buffer = Buffer(config['buffer_size'], device)
    accuracy = []

    # Eval without training
    random_accuracy = evaluate_past(model, len(test_set) - 1, test_set, loss, device)

    for index, data_set in enumerate(train_set):
        model.train()
        print(f"----- DOMAIN {index} -----")
        print("Training model...")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)
        # domain_writer = SummaryWriter(
        #     f'./runs/continual/train/domain_{index}/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))

        for epoch in tqdm(range(config['epochs'])):
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

                y_pred = model(inputs)
                s_loss = loss(y_pred.squeeze(1), labels)
                acc = binary_accuracy(y_pred.squeeze(1), labels)
                # METRICHE INTERNE EPOCA
                epoch_loss.append(s_loss.item())
                epoch_acc.append(acc.item())

                s_loss.backward()
                optimizer.step()

                if epoch == 0:
                    buffer.add_data(examples=x.to(device), labels=y.to(device))

            global_writer.add_scalar('Train_global/Loss', statistics.mean(epoch_loss),
                                     epoch + (config['epochs'] * index))
            global_writer.add_scalar('Train_global/Accuracy', statistics.mean(epoch_acc),
                                     epoch + (config['epochs'] * index))

            # domain_writer.add_scalar(f'Train_D{index}/Loss', statistics.mean(epoch_loss), epoch)
            # domain_writer.add_scalar(f'Train_D{index}/Accuracy', statistics.mean(epoch_acc), epoch)

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.5f}')

            # Last epoch (only for stats)
            if epoch == 499:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.5f}')

        # Test on domain just trained + old domains
        accuracy.append(evaluate_past(model, index, test_set, loss, device))
        if index != len(train_set) - 1:
            # accuracy[index] = accuracy[index] + evaluate_next(model, index, test_set, loss, device)
            accuracy[index].append(evaluate_next(model, index, test_set, loss, device))

    # Check buffer distribution
    buffer.check_distribution()

    # Compute transfer metrics
    backward = backward_transfer(accuracy)
    forward = forward_transfer(accuracy, random_accuracy)
    forget = forgetting(accuracy)
    print(f'Backward transfer: {backward}')  # todo Sono % in accuracy?
    print(f'Forward transfer: {forward}')
    print(f'Forgetting: {forget}')


def train_online(data, model, loss, optimizer, epochs, device, domain, global_writer):
    """
    :param data: data
    :param model: PyTorch model
    :param loss: loss function
    :param optimizer: optimizer
    :param epochs: training epochs
    :param device: device cuda/cpu
    :param domain: domain
    :param global_writer: global SummaryWriter for tensorboard
    """
    print("Training model...")
    model.train()

    # domain_writer = SummaryWriter(
    #     f'./runs/online/train/domain_{domain}/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))

    for i in tqdm(range(epochs)):
        epoch_loss = []
        epoch_acc = []
        for j, (x, y) in enumerate(data):
            optimizer.zero_grad()

            x = x.to(device)
            y_pred = model(x)
            y = y.to(device)
            s_loss = loss(y_pred.squeeze(1), y)
            acc = binary_accuracy(y_pred.squeeze(1), y)

            epoch_loss.append(s_loss.item())
            epoch_acc.append(acc.item())

            s_loss.backward()
            optimizer.step()

        global_writer.add_scalar('Train_global/Loss', statistics.mean(epoch_loss), i + (epochs * domain))
        global_writer.add_scalar('Train_global/Accuracy', statistics.mean(epoch_acc), i + (epochs * domain))

        # domain_writer.add_scalar(f'Train_D{domain}/Loss', statistics.mean(epoch_loss), i)
        # domain_writer.add_scalar(f'Train_D{domain}/Accuracy', statistics.mean(epoch_acc), i)

        if i % 100 == 0:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')
        # Last epoch (only for stats)
        if i == 499:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f}'
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

    torch.save(model.state_dict(), f'checkpoints/online/model_d{domain}.pt')

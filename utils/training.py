import torch
import statistics
import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.buffer import Buffer
from utils.utils import binary_accuracy
from torch.utils.tensorboard import SummaryWriter


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

    print(f"Test error: {statistics.mean(test_loss)} | Test accuracy: {statistics.mean(test_acc):.5f}")


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
    buffer = Buffer(config['buffer_size'], device)
    for index, data_set in enumerate(train_set):  # PER OGNI TASK
        model.train()
        print(f"----- DOMAIN {index} -----")
        # train(model, loss, config['batch_size'], data_set, config['epochs'], optimizer, buffer, device)
        print("Training model...")
        train_loader = DataLoader(data_set, batch_size=config['batch_size'], shuffle=False)
        domain_acc = []
        domain_loss = []

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

            if epoch % 100 == 0:
                print(f'\nEpoch {epoch:03}/{config["epochs"]} | Loss: {statistics.mean(epoch_loss):.5f} '
                      f'| Acc: {statistics.mean(epoch_acc):.5f}')

            # METRICHE DEL DOMINIO
            domain_acc.append(statistics.mean(epoch_acc))
            domain_loss.append(statistics.mean(epoch_loss))

        print(f"Training loss: {statistics.mean(domain_loss):.5f} |"
              f" Training acc: {statistics.mean(domain_acc):.5f}")
        # Test on domain just trained + old domains
        for past in range(index + 1):
            print(f"Testing model on domain {past}")
            test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)  # DIM ORIGINALE SENZA REPLAY
            test(model, loss, test_loader, device)

    # Test to check buffer distribution
    buffer.check_distribution()


def train_online(data, model, loss, optimizer, epochs, device, domain):
    """
    :param data: data
    :param model: PyTorch model
    :param loss: loss function
    :param optimizer: optimizer
    :param epochs: training epochs
    :param device: device cuda/cpu
    :param domain: domain
    """
    print("Training model...")
    model.train()

    # TODO better same plot for all domains or one for each?
    writer = SummaryWriter('./runs/online/train/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))

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

        writer.add_scalar('Train/Loss', statistics.mean(epoch_loss), i)
        writer.add_scalar('Train/Accuracy', statistics.mean(epoch_acc), i)
        if i % 100 == 0:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

    torch.save(model.state_dict(), f'checkpoints/online/model_d{domain}.pt')

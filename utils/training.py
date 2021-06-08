import torch
import statistics
import datetime

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.buffer import Buffer
from utils.utils import binary_accuracy, check_buffer_distribution
from torch.utils.tensorboard import SummaryWriter


def train(model, loss, batch_size, data_set, epochs, optimizer, index, buffer, device):
    print("Training model...")
    train_loader = DataLoader(data_set, batch_size=batch_size,
                              shuffle=False)  # DIM ORIGINALE SENZA REPLAY
    domain_acc = []
    domain_loss = []

    for epoch in tqdm(range(epochs)):
        epoch_loss = []
        epoch_acc = []
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = x.to(device)
            labels = y.to(device)
            # inputs = x.cuda()
            # labels = y.cuda()
            if index > 0:
                buf_input, buf_label = buffer.get_data(batch_size)  # Strategy 50/50
                # CONVERTIREV LIST IN TENSOR FLOAT
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
            # buffer.add_data(examples=x.cuda(), labels=y.cuda())

        if epoch % 100 == 0:
            print(f'\nEpoch {epoch:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

        # METRICHE DEL DOMINIO
        domain_acc.append(statistics.mean(epoch_acc))
        domain_loss.append(statistics.mean(epoch_loss))

    torch.save(model.state_dict(), 'model.pt')
    print(f"Training loss: {statistics.mean(domain_loss):.5f} |"
          f" Training acc: {statistics.mean(domain_acc):.5f}")


def test(model, loss, test_loader, device):
    print("Testing model...")
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    test_loss = []
    test_acc = []
    predicted = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            # x = x.cuda()
            # y = y.cuda()
            pred = model(x)
            s_loss = loss(pred.squeeze(1), y)
            acc = binary_accuracy(pred.squeeze(1), y)
            predicted.append(pred.cpu().numpy().item())
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Test error: {statistics.mean(test_loss)} | Test accuracy: {statistics.mean(test_acc):.5f}")


def train_cl(train_set, test_set, model, loss, optimizer, device, config):
    buffer = Buffer(config['buffer_size'], device)
    # train_loss = []
    # train_acc = []
    for index, data_set in enumerate(train_set):  # PER OGNI TASK
        model.train()
        print(f"----- DOMAIN {index} -----")
        if index == 0:
            train(model, loss, config['batch_size'], data_set, config['epochs'], optimizer, index, buffer, device)
        else:
            train(model, loss, config['batch_size'] // 2, data_set, config['epochs'], optimizer, index, buffer, device)

        test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)  # DIM ORIGINALE SENZA REPLAY
        test(model, loss, test_loader, device)

    # Test to check buffer distribuition
    check_buffer_distribution(buffer)


def train_online(data, model, loss, optimizer, epochs, device):
    # bce = []
    # accuracy = []
    model.train()

    writer = SummaryWriter('./runs/online/train/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))

    for i in tqdm(range(epochs)):
        epoch_loss = []
        epoch_acc = []
        for j, (x, y) in enumerate(data):
            optimizer.zero_grad()

            x = x.to(device)
            # x = x.cuda()
            y_pred = model(x)

            y = y.to(device)
            # y = y.cuda()
            s_loss = loss(y_pred.squeeze(1), y)
            acc = binary_accuracy(y_pred.squeeze(1), y)
            epoch_loss.append(s_loss.item())
            epoch_acc.append(acc.item())

            s_loss.backward()
            optimizer.step()

        writer.add_scalar('Train/Loss', statistics.mean(epoch_loss), i)
        writer.add_scalar('Train/Accuracy', statistics.mean(epoch_acc), i)
        if i % 50 == 0:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

        # bce.append(statistics.mean(epoch_loss))
        # accuracy.append(statistics.mean(epoch_acc))

    # plt.plot(bce, label='BCE')
    # plt.autoscale(axis='x', tight=True)
    # title = " Train BCE domain " + str(index)
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.show()
    #
    # plt.plot(accuracy, label='accuracy')
    # plt.autoscale(axis='x', tight=True)
    # title = "Train Accuracy domain " + str(index)
    # plt.title(title)
    # plt.legend(loc='best')
    # plt.show()

    torch.save(model.state_dict(), 'model.pt')

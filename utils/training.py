import torch
import statistics
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.buffer import Buffer
from utils.utils import binary_accuracy


def train(model, loss, batch_size, data_set, epochs, optimizer, index, buffer):
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

            inputs = x.cuda()
            labels = y.cuda()
            if index > 0:
                buf_input, buf_label = buffer.get_data(batch_size)  # Strategy 50/50
                # CONVERTIREV LIST IN TENSOR FLOAT
                inputs = torch.cat((inputs, buf_input))
                labels = torch.cat((labels, buf_label))

            y_pred = model(inputs)
            s_loss = loss(y_pred.squeeze(1), labels)
            acc = binary_accuracy(y_pred.squeeze(1), labels)
            # METRICHE INTERNE EPOCA
            epoch_loss.append(s_loss.item())
            epoch_acc.append(acc.item())

            s_loss.backward()
            optimizer.step()
            buffer.add_data(examples=x.cuda(), labels=y.cuda())

        if epoch % 100 == 0:
            print(f'\nEpoch {epoch:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

        # METRICHE DEL DOMINIO
        domain_acc.append(statistics.mean(epoch_acc))
        domain_loss.append(statistics.mean(epoch_loss))

    torch.save(model.state_dict(), 'model.pt')
    print(f"Training loss: {statistics.mean(domain_loss):.5f} |"
          f" Training acc: {statistics.mean(domain_acc):.5f}")


def test(model, loss, test_loader):
    print("Testing model...")
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    test_loss = []
    test_acc = []
    predicted = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            s_loss = loss(pred.squeeze(1), y)
            acc = binary_accuracy(pred.squeeze(1), y)
            predicted.append(pred.cpu().numpy().item())
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Test error: {statistics.mean(test_loss)} | Test accuracy: {statistics.mean(test_acc):.5f}")


def train_cl(train_set, test_set, model, loss, optimizer,device,config):
    buffer = Buffer(config['buffer_size'], device)
    # train_loss = []
    # train_acc = []
    for index, data_set in enumerate(train_set):  # PER OGNI TASK
        model.train()
        print(f"----- DOMAIN {index} -----")
        if index == 0:
            train(model, loss, config['batch_size'], data_set, config['epochs'], optimizer, index, buffer)
        else:
            train(model, loss, config['batch_size'] // 2, data_set, config['epochs'], optimizer, index, buffer)

        test_loader = DataLoader(test_set[index], batch_size=1, shuffle=False)  # DIM ORIGINALE SENZA REPLAY
        test(model, loss, test_loader)


def train_online(data, model, loss, optimizer, epochs, index):
    bce = []
    accuracy = []
    model.train()

    for i in tqdm(range(epochs)):
        epoch_loss = []
        epoch_acc = []
        for j, (x, y) in enumerate(data):
            optimizer.zero_grad()

            x = x.cuda()
            y_pred = model(x)

            y = y.cuda()
            s_loss = loss(y_pred.squeeze(1), y)
            acc = binary_accuracy(y_pred.squeeze(1), y)
            epoch_loss.append(s_loss.item())
            epoch_acc.append(acc.item())

            s_loss.backward()
            optimizer.step()

        if i % 50 == 0:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(epoch_loss):.5f} '
                  f'| Acc: {statistics.mean(epoch_acc):.5f}')

        bce.append(statistics.mean(epoch_loss))
        accuracy.append(statistics.mean(epoch_acc))

    plt.plot(bce, label='BCE')
    plt.autoscale(axis='x', tight=True)
    title = " Train BCE domain " + str(index)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

    plt.plot(accuracy, label='accuracy')
    plt.autoscale(axis='x', tight=True)
    title = "Train Accuracy domain " + str(index)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

    torch.save(model.state_dict(), 'model.pt')

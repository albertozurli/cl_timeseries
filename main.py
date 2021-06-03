import argparse
import os
import statistics
import warnings

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

import detection.sdt.changepoint as detection
from utils.models import RegressionMLP, ClassficationMLP
from utils.utils import read_csv, binary_accuracy, split_train_test, compute_diff

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'

parser = argparse.ArgumentParser(description='Thesis')
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size dimension (default: 64)")
parser.add_argument('--epochs', type=int, default=500,
                    help="Number of train epochs (default: 500)")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="Learning rate (default: 0.0001)")
parser.add_argument('--filename', type=str, default="brent-monthly.csv",
                    help="CSV file(default: brent-monthly.csv)")
parser.add_argument('--train', action='store_true',
                    help="Train the model")
parser.add_argument('--test', action='store_true',
                    help="Test the model")
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--processing', default='none', choices=['none', 'difference'],
                    help="Type of pre-processing")


def train_model(data, model, loss, optimizer, epochs, index):
    bce = []
    accuracy = []
    model.train()

    train_writer = SummaryWriter('./runs/' + 'train/' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M'))

    for i in tqdm(range(epochs)):
        loss_list = []
        acc_list = []
        for j, (x, y) in enumerate(data):
            optimizer.zero_grad()

            x = x.cuda()
            y_pred = model(x)

            y = y.cuda()
            s_loss = loss(y_pred.squeeze(1), y)
            acc = binary_accuracy(y_pred.squeeze(1), y)
            loss_list.append(s_loss.item())
            acc_list.append(acc.item())

            s_loss.backward()
            optimizer.step()

            train_writer.add_scalar('BCE/Train', s_loss, i)
            train_writer.add_scalar('Accuracy/Train', acc, i)

        if i % 50 == 0:
            print(f'\nEpoch {i:03}/{epochs} | Loss: {statistics.mean(loss_list):.5f} '
                  f'| Acc: {statistics.mean(acc_list):.5f}')

        bce.append(statistics.mean(loss_list))
        accuracy.append(statistics.mean(acc_list))

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


def test_model(data, model, loss, index):
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    test_loss = []
    accuracy = []
    predicted = []

    writer = SummaryWriter('./runs/' + 'test/' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M'))

    for j, (x, y) in enumerate(data):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            s_loss = loss(pred.squeeze(1), y)
            acc = binary_accuracy(pred.squeeze(1), y)
            predicted.append(pred.cpu().numpy().item())
            accuracy.append(acc.item())
            test_loss.append(s_loss.item())

            writer.add_scalar('BCE/Test', s_loss.item(), j)
            writer.add_scalar('Accuracy/Test', acc.item(), j)

    print(f"Test error: {statistics.mean(test_loss)} | Test accuracy: {statistics.mean(accuracy):.5f}")

    # Scatter plot
    tmp = predicted[1:]
    tmp.append(0)
    plt.scatter(predicted, tmp)
    title = "Scatter plot domain " + str(index)
    plt.title(title)
    plt.show()


def main(config):
    raw_data = read_csv(config["filename"])

    # Online changepoint
    det = detection.BayesOnline()
    chp_online = det.find_changepoints(raw_data, past=50, prob_threshold=0.3)
    chps = chp_online[1:]

    # print(chp_online)
    # plt.figure(figsize=(10, 6))
    # plt.plot(raw_data)
    # for i in chps:
    #     plt.axvline(i, color="red")
    # plt.title("Bayesian Online")
    # plt.show()

    # Test grafico per valutare gli split
    # test,train = split(raw_data,chps)

    # if config['processing'] == 'none': Default everytime
    raw_data = numpy.array(raw_data).reshape(-1, 1)

    if config['processing'] == 'difference':
        raw_data = compute_diff(raw_data)

    # Split in N train/test set (data + domain feature)
    train_data, test_data = split_train_test(raw_data, chps, 4)

    # Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(0))

    # Setup and train the model
    if config["regression"]:
        model = RegressionMLP(input_size=5)
        model = model.cuda()
        loss = nn.MSELoss()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["lr"])
    else:
        model = ClassficationMLP(input_size=5)
        model = model.cuda()
        loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epochs = config["epochs"]
    print(model)

    if config["online"]:
        if config["train"]:
            for index, train_set in enumerate(train_data):
                print(f"---------- DOMAIN {index} ----------")
                print("Training")
                train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
                if index != 0:
                    model.load_state_dict(torch.load('model.pt'))
                train_model(train_loader, model=model, loss=loss, optimizer=optimizer, epochs=epochs, index=index)
                print("\nTesting")
                test_loader = DataLoader(test_data[index], batch_size=1, shuffle=False)
                test_model(test_loader, model=model, loss=loss, index=index)

        if config["test"]:
            print("-------------------")
            print("FINAL TEST EVAL")
            for idx, test_set in enumerate(test_data):
                print(f"\n DOMAIN {idx}")
                test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
                test_model(test_loader, model=model, loss=loss, index=idx)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

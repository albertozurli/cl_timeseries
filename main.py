import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import argparse

import detection.sdt.changepoint as detection

from models import RegressionMLP,ClassficationMLP
from utils import printProgressBar,read_csv,binary_accuracy,split_train_test,compute_diff,split
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"]='True'

parser = argparse.ArgumentParser(description='Thesis')
parser.add_argument('--batch_size',type=int, default=128,
                    help="Batch size dimension (default: 64)")
parser.add_argument('--epochs',type=int, default=500,
                    help="Number of train epochs (default: 500)")
parser.add_argument('--lr',type=float, default=0.01,
                    help="Learning rate (default: 0.01)")
parser.add_argument('--filename',type=str, default="test-brent.csv",
                    help="CSV file(default: test-brent.csv)")
parser.add_argument('--train',action='store_true',
                    help="Train the model")
parser.add_argument('--test',action='store_true',
                    help="Test the model")
parser.add_argument('--regression',action='store_true',
                    help="Regression task")
parser.add_argument('--normalize',action='store_true',
                    help="Normalize dataset")
parser.add_argument('--standardize',action='store_true',
                    help="Normalize dataset")
parser.add_argument('--difference',action='store_true',
                    help="Normalize dataset")
parser.add_argument('--online',action='store_true',
                    help="Online Learning")
parser.add_argument('--joint',action='store_true',
                    help="Joint Learning")


def train_model(data,model,loss,optimizer,epochs):
    print("Preparing data...")
    bce = []
    accuracy = []
    model.train()
    for i in range(epochs+1):
        loss_list = []
        acc_list = []
        # printProgressBar(i + 1,epochs, prefix='Progress:', suffix='Complete', length=50)
        for j,(x,y) in enumerate(data):
            optimizer.zero_grad()

            x = x.cuda()
            y_pred = model(x)

            y = y.cuda()
            s_loss = loss(y_pred.squeeze(1), y)
            acc = binary_accuracy(y_pred.squeeze(1),y)
            loss_list.append(s_loss.item())
            acc_list.append(acc.item())

            s_loss.backward()
            optimizer.step()

        if i % 50 == 0:
            print(f'Epoch {i:03}/{epochs} | Loss: {statistics.mean(loss_list):.5f} | Acc: {statistics.mean(acc_list):.3f}')

        bce.append(statistics.mean(loss_list))
        accuracy.append(statistics.mean(acc_list))

    plt.plot(bce,label='BCE')
    plt.autoscale(axis='x', tight=True)
    plt.autoscale(axis='y', tight=True)
    plt.legend(loc='best')
    plt.show()

    plt.plot(accuracy, label='accuracy')
    plt.autoscale(axis='x', tight=True)
    plt.autoscale(axis='y', tight=True)
    plt.legend(loc='best')
    plt.show()

    torch.save(model.state_dict(),'model.pt')


def test_model(data,model,loss):
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    test_loss = []
    accuracy = []
    predicted = []
    for j,(x,y) in enumerate(data):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            s_loss = loss(pred.squeeze(1), y)
            acc = binary_accuracy(pred.squeeze(1), y)
            predicted.append(pred.cpu().numpy().item())
            accuracy.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Test error: {statistics.mean(test_loss)} | Test accuracy: {statistics.mean(accuracy)}")

    return predicted


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
    # plt.title("Bayesian Online V1")
    # plt.show()

    # Test grafico per valutare gli split
    #test,train = split(raw_data,chps)

    raw_data = numpy.array(raw_data).reshape(-1, 1)

    if config['normalize']:
        scaler = MinMaxScaler()
        raw_data = scaler.fit_transform(raw_data)
    elif config['standardize']:
        scaler = StandardScaler()
        raw_data = scaler.fit_transform(raw_data)
    elif config['difference']:
        raw_data = compute_diff(raw_data)

    # Split in N train/test set (data + domain feature)
    train_data,test_data = split_train_test(raw_data,chps,4)
    train_data = [item for sublist in train_data for item in sublist]
    test_data = [item for sublist in test_data for item in sublist]

    # Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f"Device: {device}")
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

    # Joint training
    if config["joint"]:
        if config["train"]:
            train_loader = DataLoader(train_data,batch_size=config["batch_size"],shuffle=True)
            train_model(train_loader,model=model,loss=loss,optimizer=optimizer,epochs=epochs)
        # Test phase
        if config["test"]:
            test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
            predicted = test_model(test_loader,model=model,loss=loss)
            # Scatter plot
            # tmp = predicted[1:]
            # tmp.append(0)
            # plt.scatter(predicted,tmp)
            # plt.show()

    if config["online"]:
        if config["train"]:
            for index,train_set in enumerate(train_data):
                print(f"---------- DOMAIN {index} ----------")
                print("Training\n")
                train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
                if index !=0:
                    model.load_state_dict(torch.load('model.pt'))
                train_model(train_loader, model=model, loss=loss, optimizer=optimizer, epochs=epochs)
                print("\nTesting\n")
                test_loader = DataLoader(test_data[index], batch_size=1, shuffle=True)
                predicted = test_model(test_loader,model=model,loss=loss)

        if config["test"]:
            print("FINAL TEST EVAL")
            for idx,test_set in enumerate(test_data):
                print(f"\n DOMAIN {idx}")
                test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
                predicted = test_model(test_loader, model=model, loss=loss)


if __name__=="__main__":
    args = vars(parser.parse_args())
    main(args)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import statistics
import numpy as np
import os
import argparse

from models import SimpleMLP
from utils import split_seq,printProgressBar,read_csv
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"]='True'
parser = argparse.ArgumentParser(description='Thesis')

parser.add_argument('--hidden_size', type=int, default=200, metavar='HIDDENSIZE',
                    help="Number of neurons in hidden layer (default: 200)")
parser.add_argument('--batch_size',type=int, default=128, metavar='BATCHSIZE',
                    help="Batch size dimension (default: 128)")
parser.add_argument('--epochs',type=int, default=500, metavar='EPOCHS',
                    help="Number of train epochs (default: 500)")
parser.add_argument('--lr',type=float, default=0.01, metavar='LR',
                    help="Learning rate (default: 0.01)")
parser.add_argument('--filename',type=str, default="test-brent.csv", metavar='FILENAME',
                    help="CSV file(default: test-brent.csv)")
parser.add_argument('--train',type=int, default=0, metavar='TRAIN',
                    help="if 1 train the model, 0 otherwise (default: 0)")
parser.add_argument('--test',type=int, default=0, metavar='TEST',
                    help="if 1 train the model, 0 otherwise (default: 0)")


def train_model(data,model,loss,optimizer,epochs):
    print("Preparing data...")
    mse = []
    model.train()
    for i in range(epochs+1):
        loss_list = []
        # printProgressBar(i + 1,epochs, prefix='Progress:', suffix='Complete', length=50)
        for j,(seq, label) in enumerate(data):
            optimizer.zero_grad()

            seq = seq.cuda()
            y_pred = model(seq)

            label = label.cuda()
            s_loss = loss(y_pred.squeeze(), label)
            loss_list.append(s_loss.item())
            s_loss.backward()

            optimizer.step()

        if i % 50 == 0:
            print(f'epoch: {i:3}/{epochs} loss: {statistics.mean(loss_list)}')

        mse.append(statistics.mean(loss_list))

    plt.plot(mse,label='MSE')
    plt.autoscale(axis='x', tight=True)
    plt.legend(loc='best')
    plt.show()

    torch.save(model.state_dict(),'model.pt')


def test_model(data,model,loss):
    model.load_state_dict(torch.load('model.pt'))

    model.eval()
    predicted = []
    test_loss = []
    for j,(seq, label) in enumerate(data):
        with torch.no_grad():
            seq = seq.cuda()
            label = label.cuda()
            pred = model(seq)
            s_loss = loss(pred[0], label)
            test_loss.append(s_loss.item())
            predicted.append(pred[0].cpu().numpy())
            # print(f"Predicted: {pred}, loss: {s_loss.item()}")

    print(f"Test error: {statistics.mean(test_loss)}")

    return predicted


def main(config):
    raw_data = read_csv(config["filename"])
    raw_seq = torch.Tensor(raw_data).view(-1)
    data = split_seq(raw_seq,4)
    train_data = data[:1400]
    test_data = data[1400:]
    # for seq,label in data:
    #     print(seq,label)

    # Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (f"Device: {device}")
    print(torch.cuda.get_device_name(0))

    # Setup and train the model
    model = SimpleMLP(input_size=4,hid_size=config["hidden_size"])
    model = model.cuda()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config["lr"])
    epochs = config["epochs"]
    print(model)

    if config["train"]:
        train_loader = DataLoader(train_data,batch_size=config["batch_size"],shuffle=True)
        train_model(train_loader,model=model,loss=loss,optimizer=optimizer,epochs=epochs)

    # Test phase
    if config["test"]:
        test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=True)
        predicted = test_model(test_data,model=model,loss=loss)

        plt.plot(raw_seq, label='Ground truth')
        x = np.arange(1400, 1734, 1)
        plt.plot(x, predicted, label='Predicted')
        plt.autoscale(axis='x', tight=True)
        plt.legend(loc='best')
        plt.show()

        plt.plot(x, raw_seq[-334:], label='Ground truth')
        plt.plot(x, predicted, label='Predicted')
        plt.autoscale(axis='x', tight=True)
        plt.legend(loc='best')
        plt.show()





if __name__=="__main__":
    args = vars(parser.parse_args())
    main(args)
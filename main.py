import argparse
import os
import warnings
import numpy
import torch

import torch.nn as nn
import detection.sdt.changepoint as detection

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from torch.utils.data import DataLoader
from utils.training import test, train_online, train_cl
from utils.models import RegressionMLP, ClassficationMLP
from utils.utils import read_csv, split_train_test, compute_diff

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
parser.add_argument('--buffer_size', type=int, default=500,
                    help="Size of the buffer for ER (default: 500")
parser.add_argument('--train', action='store_true',
                    help="Train the model")
parser.add_argument('--test', action='store_true',
                    help="Test the model")
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--continual', action='store_true',
                    help="Continual Learning with ER")
parser.add_argument('--processing', default='none', choices=['none', 'difference'],
                    help="Type of pre-processing")


# TODO SISTEMARE IL FORMATO NP.ARRAY/LIST/TENSOR PER IL BUFFER

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
        model = model.to(device)
        # model = model.cuda()
        loss = nn.MSELoss()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["lr"])
    else:
        model = ClassficationMLP(input_size=5)
        model = model.to(device)
        # model = model.cuda()
        loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epochs = config["epochs"]
    print(model)

    if config["online"]:
        if config["train"]:
            for index, train_set in enumerate(train_data):
                print(f"----- DOMAIN {index} -----")
                train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
                if index != 0:
                    model.load_state_dict(torch.load('model.pt'))
                train_online(train_loader, model=model, loss=loss, optimizer=optimizer, epochs=epochs,
                             device=device)
                # test domain just trained
                test_loader = DataLoader(test_data[index], batch_size=1, shuffle=False)
                test(model=model, loss=loss, test_loader=test_loader, device=device)

        if config["test"]:
            print("-------------------")
            for idx, test_set in enumerate(test_data):
                print(f"\n DOMAIN {idx}")
                test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
                test(model=model, loss=loss, test_loader=test_loader, device=device)

    if config['continual']:
        train_cl(train_set=train_data, test_set=test_data, model=model, loss=loss,
                 optimizer=optimizer, config=config, device=device)
        print("-------------------")
        for idx, test_set in enumerate(test_data):
            print(f"\n DOMAIN {idx}")
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
            test(model=model, loss=loss, test_loader=test_loader, device=device)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

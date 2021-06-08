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
from utils.utils import read_csv, split_train_test, compute_diff, eval_bayesian

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
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--continual', action='store_true',
                    help="Continual Learning with ER")
parser.add_argument('--processing', default='none', choices=['none', 'difference'],
                    help="Type of pre-processing")
parser.add_argument('--split', action='store_true',
                    help="Show domain split")


def main(config):
    raw_data = read_csv(config["filename"])

    # Online changepoint
    det = detection.BayesOnline()
    chp_online = det.find_changepoints(raw_data, past=50, prob_threshold=0.3)
    chps = chp_online[1:]

    # Evaluation bayesian analysis
    if config['split']:
        eval_bayesian(chps, raw_data)

    # if config['processing'] == 'none': Default
    raw_data = numpy.array(raw_data).reshape(-1, 1)

    if config['processing'] == 'difference':
        raw_data = compute_diff(raw_data)

    # Split in N train/test set (data + feature)
    train_data, test_data = split_train_test(raw_data, chps, 4)

    # Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(0))

    # Setup the model
    if config["regression"]:
        model = RegressionMLP(input_size=5)
        model = model.to(device)
        loss = nn.MSELoss()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["lr"])
    else:
        model = ClassficationMLP(input_size=5)
        model = model.to(device)
        loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epochs = config["epochs"]
    print(model)

    # Online training
    if config["online"]:
        for index, train_set in enumerate(train_data):
            print(f"----- DOMAIN {index} -----")
            train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
            train_online(train_loader, model=model, loss=loss, optimizer=optimizer, epochs=epochs,
                         device=device, domain=index)
            # Test on domain just trained + old domains
            for past in range(index+1):
                print(f"Testing model on domain {past}")
                test_loader = DataLoader(test_data[past], batch_size=1, shuffle=False)
                test(model=model, loss=loss, test_loader=test_loader, device=device)

    # Continual learning with ER
    if config['continual']:
        train_cl(train_set=train_data, test_set=test_data, model=model, loss=loss,
                 optimizer=optimizer, config=config, device=device)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

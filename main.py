import argparse
import os
import warnings
import torch
import datetime
import utils.training

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from torch.utils.data import DataLoader
from utils.models import RegressionMLP, ClassficationMLP
from utils.utils import read_csv, split_data, split_with_indicators, compute_diff, eval_bayesian, check_changepoints,\
    timeperiod
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch.nn as nn
import detection.sdt.changepoint as detection

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
parser.add_argument('--filename', type=str, default="oil-daily.csv",
                    help="CSV file")
parser.add_argument('--buffer_size', type=int, default=500,
                    help="Size of the buffer for ER (default: 500")
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--continual', action='store_true',
                    help="Continual Learning with ER")
parser.add_argument('--processing', default='none', choices=['none', 'difference', 'indicators'],
                    help="Type of pre-processing")
parser.add_argument('--split', action='store_true',
                    help="Show domain split")


def main(config):
    raw_data = read_csv(config["filename"])

    # Check if chps are already saved
    saved, chps = check_changepoints(config["filename"])

    # Online changepoint
    if not saved:
        det = detection.BayesOnline()
        chp_online = det.find_changepoints(raw_data, past=50, prob_threshold=0.3)
        chps = chp_online[1:]

    # Evaluation bayesian analysis
    if config['split']:
        eval_bayesian(chps, raw_data)

    # Type of dataset (yearly,quarterly...)
    n_step = timeperiod(config['filename'])

    # Split in N train/test set (data + feature)
    if config['processing'] == 'indicators':
        train_data, test_data = split_with_indicators(raw_data, chps, n_step)

    elif config['processing'] == 'difference':
        raw_data = compute_diff(raw_data)
        train_data, test_data = split_data(raw_data, chps, n_step)

    else:
        raw_data = np.array(raw_data).reshape(-1, 1)
        train_data, test_data = split_data(raw_data, chps, n_step)

    # Cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(torch.cuda.get_device_name(0))

    # Setup the model
    input_size = train_data[0][0][0].size()[0]
    if config["regression"]:
        model = RegressionMLP(input_size=input_size)
        model = model.to(device)
        loss = nn.MSELoss()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=config["lr"])
    else:
        model = ClassficationMLP(input_size=input_size)
        model = model.to(device)
        loss = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    epochs = config["epochs"]
    print(model)

    # Online training
    if config["online"]:
        global_writer = SummaryWriter('./runs/online/train/global/' + datetime.datetime.now().strftime('%m_%d_%H_%M'))
        for index, train_set in enumerate(train_data):
            print(f"----- DOMAIN {index} -----")
            train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
            utils.training.train_online(train_loader, model=model, loss=loss, optimizer=optimizer, epochs=epochs,
                                        device=device, domain=index, global_writer=global_writer)
            # Test on domain just trained + old domains
            for past in range(index + 1):
                print(f"Testing model on domain {past}")
                test_loader = DataLoader(test_data[past], batch_size=1, shuffle=False)
                utils.training.test(model=model, loss=loss, test_loader=test_loader, device=device)

    # Continual learning with ER
    if config['continual']:
        utils.training.train_cl(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                optimizer=optimizer, config=config, device=device)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

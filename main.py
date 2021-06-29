import argparse
import os
import warnings
import torch

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from utils.models import RegressionMLP, ClassficationMLP
from utils.training import train_er, train_online
from utils.utils import read_csv, split_data, split_with_indicators, compute_diff, eval_bayesian, check_changepoints, \
    timeperiod

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
parser.add_argument('--filename', type=str, help="CSV file")
parser.add_argument('--buffer_size', type=int, default=500,
                    help="Size of the buffer for ER (default: 500")
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--er', action='store_true',
                    help="Continual Learning with ER")
parser.add_argument('--processing', default='none', choices=['none', 'difference', 'indicators'],
                    help="Type of pre-processing")
parser.add_argument('--split', action='store_true',
                    help="Show domain split")
parser.add_argument('--suffix', type=str, default="",
                    help="Suffix name")


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
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'checkpoints/model_scratch.pt')

    print(model)

    if not config['suffix']:
        suffix = config['filename'].partition('-')[0]
    else:
        suffix = config['suffix']

    # Online training
    if config["online"]:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        train_online(train_set=train_data, test_set=test_data, model=model, loss=loss,
                     optimizer=optimizer, config=config, device=device, suffix=suffix)
    # Continual learning with ER
    if config['er']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        train_er(train_set=train_data, test_set=test_data, model=model, loss=loss,
                 optimizer=optimizer, device=device, config=config, suffix=suffix)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

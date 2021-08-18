import argparse
import os
import warnings
import torch
import time

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from utils.models import RegressionMLP, ClassficationMLP, SimpleCNN
import utils.training
from utils.utils import read_csv, split_data, split_with_indicators, eval_bayesian, check_changepoints, \
    timeperiod
from torchsummary import summary

import numpy as np
import torch.nn as nn
import detection.sdt.changepoint as detection

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'

# General Paramteres
parser = argparse.ArgumentParser(description='Thesis')
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size dimension")
parser.add_argument('--epochs', type=int, default=300,
                    help="Number of train epochs")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="Learning rate")
parser.add_argument('--dataset', type=str, help="CSV file")
parser.add_argument('--regression', action='store_true',
                    help="Regression task")
parser.add_argument('--processing', default='none', choices=['none', 'difference', 'indicators'],
                    help="Type of pre-processing")
parser.add_argument('--split', action='store_true',
                    help="Show domain split")
parser.add_argument('--suffix', type=str, default="",
                    help="Suffix name")
parser.add_argument('--evaluate', action='store_true',
                    help="Test previous + current domain each epoch")
# Network
parser.add_argument('--cnn', action='store_true',
                    help="Convolutional Network")
parser.add_argument('--dropout', type=float, default=0.5,
                    help="Probabilty for dropout in MLP")
parser.add_argument('--l1_lambda', type=float, default=0.001,
                    help="Regularization param in L1 Norm (CNN only)")
# Methods
parser.add_argument('--online', action='store_true',
                    help="Online Learning")
parser.add_argument('--er', action='store_true',
                    help="Continual Learning with ER")
parser.add_argument('--der', action='store_true',
                    help="Continual Learning with Dark ER")
parser.add_argument('--ewc', action='store_true',
                    help="Continual Learning with EWC")
parser.add_argument('--si', action='store_true',
                    help="Continual Learning with SI")
parser.add_argument('--gem', action='store_true',
                    help="Continual Learning with GEM")
parser.add_argument('--agem', action='store_true',
                    help="Continual Learning with A-GEM")
parser.add_argument('--agem_r', action='store_true',
                    help="Continual Learning with A-GEM with Reservoir")
# EWC Parameters
parser.add_argument('--gamma', type=float, default=0.7,
                    help="gamma value for EWC")
parser.add_argument('--e_lambda', type=float, default=100,
                    help="lambda value for EWC")
# SI Parameters
parser.add_argument('--xi', type=float, default=1,
                    help="xi value for SI")
parser.add_argument('--c', type=float, default=0.5,
                    help="c value for SI")
# ER/DER Parameters
parser.add_argument('--buffer_size', type=int, default=500,
                    help="Size of the buffer for ER/DER")
parser.add_argument('--alpha', type=float, default=0.1,
                    help="penalty weight for DER")
# GEM Parameters
parser.add_argument('--gem_gamma', type=float, default=0.25,
                    help="gamma value for GEM")


def main(config):
    start = time.time()

    raw_data = read_csv(config["dataset"])

    # Check if chps are already saved
    saved, chps = check_changepoints(config["dataset"])

    # Online changepoint
    if not saved:
        det = detection.BayesOnline()
        # past and th heavily depend on data
        chp_online = det.find_changepoints(raw_data, past=50, prob_threshold=0.3)
        chps = chp_online[1:]

    # Evaluation bayesian analysis
    if config['split']:
        eval_bayesian(chps, raw_data)

    # Type of dataset (yearly,quarterly...)
    n_step = timeperiod(config['dataset'])

    # Split in N train/test set (data + feature)
    if config['processing'] == 'indicators':
        train_data, test_data = split_with_indicators(config, raw_data, chps, n_step)
    elif config['processing'] == 'difference':
        raw_data = np.diff(raw_data, axis=0)
        train_data, test_data = split_data(config, raw_data, chps, n_step)
    else:
        raw_data = np.array(raw_data).reshape(-1, 1)
        train_data, test_data = split_data(config, raw_data, chps, n_step)

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
        if config["cnn"]:
            model = SimpleCNN(input_size=input_size)
        else:
            model = ClassficationMLP(input_size=input_size, dropout=config['dropout'])
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        model = model.to(device)
        loss = nn.CrossEntropyLoss()
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'checkpoints/model_scratch.pt')

    # print(model)
    summary(model, train_data[0][0][0].size())

    if not config['suffix']:
        suffix = config['dataset'].partition('-')[0]
    else:
        suffix = config['suffix']

    # Online training
    if config["online"]:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_online(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                    optimizer=optimizer, config=config, device=device, suffix=suffix)

    # Continual learning with ER
    if config['er']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_er(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with Dark ER
    if config['der']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_dark_er(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                     optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with EWC
    if config['ewc']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_ewc(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                 optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with SI
    if config['si']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_si(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with GEM
    if config['gem']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_gem(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                 optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with A-GEM
    if config['agem']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_a_gem(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                   optimizer=optimizer, device=device, config=config, suffix=suffix)

    # Continual learning with A-GEM with Reservoir
    if config['agem_r']:
        initial_model = torch.load('checkpoints/model_scratch.pt')
        model.load_state_dict(initial_model['model_state_dict'])
        optimizer.load_state_dict(initial_model['optimizer_state_dict'])
        utils.training.train_a_gem_r(train_set=train_data, test_set=test_data, model=model, loss=loss,
                                     optimizer=optimizer, device=device, config=config, suffix=suffix)
    end = time.time()
    print("\nTime elapsed: ", end - start, "s")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(args)

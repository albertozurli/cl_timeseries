import argparse
import os
import warnings
import torch
import time
import random

from models.online import train_online
from models.agem import train_agem
from models.agem_r import train_agem_r
from models.si import train_si
from models.gem import train_gem
from models.dark_exp_replay import train_dark_er
from models.exp_replay import train_er
from models.ewc import train_ewc
from models.derpp import train_derpp

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from utils.backbone import ClassficationMLP, SimpleCNN
from utils.utils import read_csv, split_data, split_with_indicators, eval_bayesian, check_changepoints, timeperiod
from torchsummary import summary

import numpy as np
import torch.nn as nn
import detection.sdt.changepoint as detection

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--epochs', type=int, default=300,
                        help="Number of train epochs")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument('--dataset', type=str, help="CSV file")
    parser.add_argument('--processing', default='none', choices=['none', 'difference', 'indicators'],
                        help="Type of pre-processing")
    parser.add_argument('--split', action='store_true',
                        help="Show tasks split")
    parser.add_argument('--suffix', type=str, default="default",
                        help="Suffix name")
    parser.add_argument('--evaluate', action='store_true',
                        help="Test current and previous tasks each epoch")
    parser.add_argument('--seed',type=int,default=230,
                        help='Seed')

    # Network arguments
    parser.add_argument('--cnn', action='store_true',
                        help="Convolutional Network")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Probability for dropout in MLP")
    parser.add_argument('--l1_lambda', type=float, default=0.01,
                        help="Regularization param in L1 Norm (used in CNN only)")
    # Methods
    parser.add_argument('--model', default='online',
                        choices=['online', 'er', 'der', 'derpp', 'ewc', 'si', 'gem', 'agem', 'agem_r'],
                        help="CL method")
    # Regularization arguments
    parser.add_argument('--gamma', type=float, default=0.7,
                        help="gamma value for EWC")
    parser.add_argument('--e_lambda', type=float, default=100,
                        help="lambda value for EWC")
    parser.add_argument('--xi', type=float, default=1,
                        help="xi value for SI")
    parser.add_argument('--c', type=float, default=0.5,
                        help="c value for SI")
    # Replay arguments
    parser.add_argument('--buffer_size', type=int, default=500,
                        help="Size of the buffer for replay methods")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="penalty weight for DER")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="penalty weight for DER++")
    parser.add_argument('--gem_gamma', type=float, default=0.25,
                        help="gamma value for GEM")

    args = vars(parser.parse_args())
    return args


def main(config):
    start = time.time()

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])

    # Read raw time series
    raw_data = read_csv(config["dataset"])

    # Check if chekpoints are already stored
    saved, chps = check_changepoints(config["dataset"])

    # Online changepoint
    if not saved:
        det = detection.BayesOnline()
        # past and threshold heavily depend on data
        chp_online = det.find_changepoints(raw_data, past=50, prob_threshold=0.2)
        chps = chp_online[1:]

    # Evaluation bayesian analysis
    if config['split']:
        eval_bayesian(chps, raw_data)

    # Type of dataset (yearly,quarterly...)
    n_step = timeperiod(config['dataset'])

    # Split in N train/test set (data + features)
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

    # Setup the backbone
    input_size = train_data[0][0][0].size()[0]
    if config["cnn"]:
        model = SimpleCNN(input_size=input_size)
    else:
        model = ClassficationMLP(input_size=input_size, dropout=config['dropout'])

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.7)
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'checkpoints/model_scratch.pt')

    # print(model)
    summary(model, train_data[0][0][0].size())

    # Online training
    if config["model"] == 'online':
        train_online(train_set=train_data, test_set=test_data, model=model, loss=loss,
                     optimizer=optimizer, config=config, device=device, suffix=config['suffix'])

    # Continual learning with ER
    if config["model"] == 'er':
        train_er(train_set=train_data, test_set=test_data, model=model, loss=loss,
                 optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with Dark ER
    if config["model"] == 'der':
        train_dark_er(train_set=train_data, test_set=test_data, model=model, loss=loss,
                      optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with Dark ER++
    if config["model"] == 'derpp':
        train_derpp(train_set=train_data, test_set=test_data, model=model, loss=loss,
                    optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with EWC
    if config["model"] == 'ewc':
        train_ewc(train_set=train_data, test_set=test_data, model=model, loss=loss,
                  optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with SI
    if config["model"] == 'si':
        train_si(train_set=train_data, test_set=test_data, model=model, loss=loss,
                 optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with GEM
    if config["model"] == 'gem':
        train_gem(train_set=train_data, test_set=test_data, model=model, loss=loss,
                  optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with A-GEM
    if config["model"] == 'agem':
        train_agem(train_set=train_data, test_set=test_data, model=model, loss=loss,
                   optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    # Continual learning with A-GEM with Reservoir
    if config["model"] == 'agem_r':
        train_agem_r(train_set=train_data, test_set=test_data, model=model, loss=loss,
                     optimizer=optimizer, device=device, config=config, suffix=config['suffix'])

    end = time.time()
    print("\nTime elapsed: ", end - start, "s")


if __name__ == "__main__":
    args = parse_args()
    main(args)

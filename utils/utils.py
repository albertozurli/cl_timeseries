import statistics
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def compute_diff(data):
    diff = np.diff(data, axis=0)
    return diff.tolist()


def split_train_test(data, chps, n_step):
    """
    :param data: data
    :param chps: list of changepoint detected
    :param n_step: size of a sequence
    :return: train set and test set divided
    """
    train_data = []
    test_data = []
    tmp = np.split(data, chps)
    for index, subdata in enumerate(tmp):
        seq = []
        i = 0
        while i < len(subdata):
            end_seq = i + n_step
            if end_seq > (len(subdata) - 4):
                break
            seq_x, y = subdata[i:end_seq], subdata[end_seq + 3]
            domain = index
            input = np.append(seq_x, domain)

            label = 0.  # Target value lower or equal than input sequence
            if y > statistics.mean(seq_x.flatten()):
                label = 1.  # Target value greater than input sequence

            input = torch.Tensor(input)
            label = torch.Tensor(np.array(label))
            seq.append((input, label))
            i += 1
        train_data.append(seq[:round(len(seq) * 0.75)])
        test_data.append(seq[round(len(seq) * 0.75):])

    return train_data, test_data


def read_csv(filename):
    """
    :param filename: name of the file
    :return: list of values extracted from the file
    """
    path = Path.cwd()
    csv_path = path.joinpath('dataset', filename)
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    value_list = lines[0].split(',')
    value_list.pop(0)
    value_list = [float(i) for i in value_list]
    return value_list


def binary_accuracy(y_pred, y_true):
    """
    :param y_pred: label predicted
    :param y_true: true label
    :return: accuracy
    """
    y_pred_tag = torch.round(y_pred)
    result_sum = (y_pred_tag == y_true).sum().float()
    acc = result_sum / y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc


def eval_bayesian(chps, raw_data):
    """
    Visual evaluation of Bayesian changepoint detection and split in different domains
    :param chps: array of changepoint
    :param raw_data:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(raw_data)
    for i in chps:
        plt.axvline(i, color="red")
    plt.title("Bayesian Online")
    plt.show()

    # Evaluation of train/test split
    splits = np.split(raw_data, chps)
    for index, subdata in enumerate(splits):
        plt.plot(subdata[:round(len(subdata) * 0.75)], color='red', label='Train')
        tmp = np.arange(round(len(subdata) * 0.75), len(subdata))
        plt.plot(tmp, subdata[round(len(subdata) * 0.75):], color='green', label='Test')
        plt.legend(loc='best')
        plt.title(f'Test/Train split domain {index}')
        plt.show()

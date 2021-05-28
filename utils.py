import os.path as osp
import statistics
import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_diff(data):
    n_data = np.asarray(data)
    diff = np.diff(n_data)
    return diff.tolist()


def split(data,chps):
    train_data = []
    test_data = []
    tmp = np.split(data,chps)
    for index,subdata in enumerate(tmp):
        train_data.append(subdata[:round(len(subdata) * 0.75)])
        test_data.append(subdata[round(len(subdata) * 0.75):])
        plt.plot(subdata[:round(len(subdata) * 0.75)])
        plt.show()
        plt.plot(subdata[round(len(subdata) * 0.75):])
        plt.show()


    train_data = [item for sublist in train_data for item in sublist]
    plt.plot(train_data)
    plt.show()
    test_data = [item for sublist in test_data for item in sublist]
    plt.plot(test_data)
    plt.show()

    return train_data,test_data


def split_train_test(data,chps,n_step):
    train_data = []
    test_data = []
    tmp = np.split(data,chps)
    for index,subdata in enumerate(tmp):
        seq = []
        i = 0
        while i < len(subdata):
            end_seq = i + n_step
            if end_seq > (len(subdata) - 4):
                break
            seq_x, y = subdata[i:end_seq], subdata[end_seq + 3]
            domain = np.array(np.digitize(index, chps))
            input = np.concatenate((seq_x, domain.reshape(1, -1))).squeeze()

            label = 0.  # Target value lower or equal than input sequence
            if y > statistics.mean(seq_x.flatten()):
                label = 1.  # Target value greater than input sequence

            input = torch.Tensor(input)
            label = torch.Tensor(np.array(label))
            seq.append((input,label))
            i += 1
        train_data.append(seq[:round(len(seq) * 0.75)])
        test_data.append(seq[round(len(seq) * 0.75):])

    return train_data,test_data


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)


def read_csv(filename):
    path="dataset/"
    csv_path = osp.join(path, filename)
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
    value_list = lines[0].split(',')
    value_list.pop(0)
    value_list = [float(i) for i in value_list]
    return value_list


def binary_accuracy(y_pred,y_true):
    y_pred_tag = torch.round(y_pred)
    result_sum = (y_pred_tag == y_true).sum().float()
    acc = result_sum/y_true.shape[0]
    acc = torch.round(acc*100)
    return acc

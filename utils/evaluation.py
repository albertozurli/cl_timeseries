import statistics

import torch
from torch.utils.data import DataLoader

from utils.utils import binary_accuracy


def test_epoch(model, test_loader, loss, device):
    model.eval()
    test_acc = []
    test_loss = []
    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            s_loss = loss(output, y.squeeze(0))
            _, pred = torch.max(output.data, 1)
            acc = binary_accuracy(pred, y.squeeze(1))
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    return test_acc, test_loss


def test(model, loss, test_loader, device):
    model.eval()
    test_loss = []
    test_acc = []

    for j, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            _, pred = torch.max(output.data, 1)
            s_loss = loss(output, y.squeeze(0))
            acc = binary_accuracy(pred, y.squeeze(1))
            test_acc.append(acc.item())
            test_loss.append(s_loss.item())

    print(f"Error: {statistics.mean(test_loss):.5f} | Acc: {statistics.mean(test_acc):.2f}%")
    return test_acc, test_loss


def evaluate_next(model, domain, test_set, loss, device):
    print("---Eval next domain---")
    test_loader = DataLoader(test_set[domain + 1], batch_size=1, shuffle=False)
    accuracy, _ = test(model, loss, test_loader, device)
    return statistics.mean(accuracy)


def evaluate_past(model, domain, test_set, loss, device):
    accs = []
    errors = []
    mean_accs = []
    mean_errors = []
    print("---Eval past domains---")
    for past in range(domain + 1):
        print(f"Domain {past} | ", end="")
        test_loader = DataLoader(test_set[past], batch_size=1, shuffle=False)
        accuracy, error = test(model, loss, test_loader, device)
        accs.append(accuracy)
        errors.append(error)
        mean_accs.append(statistics.mean(accuracy))
        mean_errors.append(statistics.mean(error))
    flat_accs = [item for sublist in accs for item in sublist]
    flat_errors = [item for sublist in errors for item in sublist]
    return flat_accs, flat_errors, mean_accs, mean_errors

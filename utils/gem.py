import numpy as np

from utils.buffer import Buffer


def save_grdient(parameters, gradient, gradient_dims):
    gradient.fill_(0.0)
    count = 0
    for p in parameters():
        if p.grad is not None:
            begin = 0 if count == 0 else sum(gradient_dims[:count])
            end = np.sum(gradient_dims[:count + 1])
            gradient[begin:end].copy_(p.grad.data.view(-1))
        count += 1


def overwrite_gradient(parameters, new, gradient_dims):
    count = 0
    for p in parameters():
        if p.grad is not None:
            begin = 0 if count == 0 else sum(gradient_dims[:count])
            end = np.sum(gradient_dims[:count + 1])
            new_grad = new[begin:end].contiguous().view(p.grad.data.size())
            p.grad.data.copy_(new_grad)
        count += 1

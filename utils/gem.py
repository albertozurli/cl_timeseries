import numpy as np
import torch
import quadprog
from torch.utils.data import DataLoader

from utils.buffer import Buffer


def save_gradient(parameters, gradient, gradient_dims):
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


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))


class GEM:
    def __init__(self,config,device,model,loss,optimizer):
        self.current_task = 0
        self.config = config
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.buffer = Buffer(self.config['buffer_size'],self.device)

        self.grad_dims = []
        for pp in self.model.parameters():
            self.grad_dims.append(pp.data.numel())

        self.grads_cs = []
        self.grads_da = torch.zeros(np.sum(self.grad_dims)).to(self.device)

    def end_task(self,dataset):
        self.current_task +=1
        self.grads_cs.append(torch.zeros(np.sum(self.grad_dims)).to(self.device))

        # Add data to the buffer
        num_samples = self.config['buffer_size'] // len(dataset)

        loader = DataLoader(dataset[(self.current_task - 1)], batch_size=num_samples, shuffle=False)
        x,y = next(iter(loader))
        self.buffer.add_data(examples=x.to(self.device),
                             task=torch.ones(num_samples,dtype=torch.long).to(self.device)*(self.current_task -1),
                             labels=y.to(self.device))





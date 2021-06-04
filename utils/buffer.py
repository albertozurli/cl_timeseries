import torch
import numpy as np
from typing import Tuple


def reservoir(seen_examples, buffer_size):
    """
    Reservoir sampling algorithm.
    :param seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current data is sampled, else -1
    """
    if seen_examples < buffer_size:
        return seen_examples

    rand = np.random.randint(0, seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.seen_examples = 0
        self.attributes = ['examples', 'labels']
        self.examples = [None] * self.buffer_size
        self.labels = [None] * self.buffer_size

    def add_data(self, examples, labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :return:
        """
        for i in range(examples.shape[0]):
            index = reservoir(self.seen_examples, self.buffer_size)
            self.seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)

    def get_data(self, size):
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :return:
        """
        if size > min(self.seen_examples, self.examples.shape[0]):
            size = min(self.seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)

        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self):
        """"
        Returns true if the buffer is empty, false otherwise
        """
        if self.seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self):
        """
        Return all the items in the memory buffer.
        :return: a tuple with all the items in the memory buffer
        """
        ret_tuple = (torch.stack([ee.cpu()
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)

    def clear(self):
        """"
        Set all tensors to None
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.seen_examples = 0

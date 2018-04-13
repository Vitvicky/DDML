import os
import random
import logging
import math
import torch
# import torch.cuda as cuda
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.

    `return`: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


class DDMLDataset(Dataset):
    """
    Implement a Dataset.
    """

    file_path = 'fashion-mnist_test.csv'
    dataset = []
    labels = set()
    with open(file_path) as f:
        for line in f:
            row = [float(_) for _ in line.split(',')]
            dataset.append((row[:-1], row[-1:]))
            labels.add(int(row[-1]))

    def __init__(self, label, size=10):
        """

        :param label: the label must in the pairs.
        :param size: size of the dataloader.
        """
        self.data = []

        if label not in DDMLDataset.labels:
            raise ValueError("Label not in the dataset.")

        self.label = label

        # if cuda.is_available():
        #     tensor = cuda.FloatTensor
        # else:
        #     tensor = FloatTensor

        tensor = FloatTensor

        while len(self.data) < size:
            while True:
                s1 = random.choice(DDMLDataset.dataset)
                s2 = random.choice(DDMLDataset.dataset)
                if int(s1[1][0]) == self.label:
                    break

            self.data.append(((tensor(s1[0]) / 255, tensor(s1[1])), (tensor(s2[0]) / 255, tensor(s2[1]))))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        # return len(self.features)
        return len(self.data)


class Net(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5.0, b=1.0, learning_rate=0.0001):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param learning_rate:
        """
        super(Net, self).__init__()

        self.layer_count = len(layer_shape)
        self.layers = []

        for _ in range(self.layer_count - 1):
            stdv = math.sqrt(6.0 / (layer_shape[_] + layer_shape[_ + 1]))
            layer = nn.Linear(layer_shape[_], layer_shape[_ + 1])
            layer.weight.data.uniform_(-stdv, stdv)
            layer.bias.data.uniform_(0, 0)
            self.layers.append(layer)
            self.add_module("layer{}".format(_), layer)

        # Softmax
        self.softmax_layer = nn.Linear(layer_shape[-1], len(DDMLDataset.labels))
        self.softmax = nn.Softmax()

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.b = b
        self.learning_rate = learning_rate
        self.gradient = []
        self.logger = logging.getLogger(__name__)

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return float(math.log(1 + math.exp(self.beta * z)) / self.beta)

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        return float(1 / (math.exp(-1 * self.beta * z) + 1))

    def _s_derivative(self, z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - self._s(z) ** 2

    def forward(self, feature):
        """
        Do forward.
        -----------
        :param feature: Variable, feature
        :return:
        """

        # project features through net
        x = feature
        for layer in self.layers:
            x = layer(x)
            x = self._s(x)

        return x

    def softmax_forward(self, feature):
        """
        Do softmax forward.
        -------------------
        :param feature: Variable, feature
        :return:
        """

        x = self(feature)
        x = self.softmax_layer(x)
        x = self.softmax(x)

        return x

    def compute_softmax_loss(self, dataloader):

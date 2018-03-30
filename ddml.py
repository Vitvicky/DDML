import os
import random
import logging
import math
import numpy as np
import torch
import torch.cuda as cuda
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

    def __init__(self, size=10):
        self._size = size
        self.data = []

        with open(self.file_path) as f:
            for line in random.sample(list(f), 2 * self._size):
                row = [float(_) for _ in line.split(',')]
                self.data.append((row[:-1], row[-1:]))

    def __getitem__(self, index):
        s1 = self.data[index]
        s2 = self.data[index + 1]

        if False:  # cuda.is_available():
            return (cuda.FloatTensor(s1[0]), cuda.FloatTensor(s1[1])), (cuda.FloatTensor(s2[0]), cuda.FloatTensor(s2[1]))
        else:
            return (FloatTensor(s1[0]) / 255, FloatTensor(s1[1])), (FloatTensor(s2[0]) / 255, FloatTensor(s2[1]))

    def __len__(self):
        # return len(self.features)
        return self._size


class Net(nn.Module):

    def __init__(self, layer_shape, beta=0.5, tao=5, lambda_=0.01, learning_rate=0.001):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param lambda_:
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

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.gradient = []
        self.logger = logging.getLogger(__name__)

    def forward(self, feature):
        """
        Do forward.
        -----------
        :param features: Variable, feature
        :return:
        """

        # project features through net
        x = feature
        for layer in self.layers:
            x = layer(x)
            x = self._s(x)

        return x

    def compute_gradient(self, dataloader):
        """
        Compute gradient.
        -----------------
        :param dataloader: DataLoader of train data pairs batch.
        :return:
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params_M = params[::2]
        params_b = params[1::2]

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        z_i_m = [[0 for m in range(self.layer_count - 1)] for _ in range(len(dataloader))]
        h_i_m = [[0 for m in range(self.layer_count)] for _ in range(len(dataloader))]
        z_j_m = [[0 for m in range(self.layer_count - 1)] for _ in range(len(dataloader))]
        h_j_m = [[0 for m in range(self.layer_count)] for _ in range(len(dataloader))]

        for index, (si, sj) in enumerate(dataloader):
            xi = Variable(si[0])
            xj = Variable(sj[0])
            h_i_m[index][0] = xi
            h_j_m[index][0] = xj
            for m in range(self.layer_count - 1):
                layer = self.layers[m]
                xi = layer(xi)
                xj = layer(xj)
                z_i_m[index][m] = xi
                z_j_m[index][m] = xj
                xi = self._s(xi)
                xj = self._s(xj)
                h_i_m[index][m + 1] = xi
                h_j_m[index][m + 1] = xj

        #
        # calculate delta_ij(m)
        # calculate delta_ji(m)
        #

        delta_ij_m = [[0 for m in range(self.layer_count - 1)] for _ in range(len(dataloader))]
        delta_ji_m = [[0 for m in range(self.layer_count - 1)] for _ in range(len(dataloader))]

        # M = layer_count - 1, then we also need to project 1,2,3 to 0,1,2
        M = self.layer_count - 1 - 1

        # calculate delta(M)
        for index, (si, sj) in enumerate(dataloader):
            xi = Variable(si[0])
            xj = Variable(sj[0])
            yi = si[1]
            yj = sj[1]

            # calculate c
            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            c = 1 - l * (self.tao - self._compute_distance(xi, xj))

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M + 1] - h_j_m[index][M + 1])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M + 1] - h_i_m[index][M + 1])) * self._s_derivative(z_j_m[index][M])

        # calculate delta(m)
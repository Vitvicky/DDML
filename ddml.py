import logging
import numpy as np
import torch
import torch.cuda as cuda
from torch import FloatTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from metric_learn import NCA


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

    def __init__(self, size=0):
        self.features = []
        self.labels = []

        with open(self.file_path) as f:
            from itertools import islice
            if size > 0:
                for line in islice(f, size):
                    row = [float(_) for _ in line.split(',')]
                    self.features.append(row[:-1])
                    self.labels.append([row[-1]])
            else:
                for line in f:
                    row = [float(_) for _ in line.split(',')]
                    self.features.append(row[:-1])
                    self.labels.append([row[-1]])

    def __getitem__(self, index):
        # return cuda.FloatTensor(self.features[index]), cuda.FloatTensor(self.labels[index])
        return FloatTensor(self.features[index]), FloatTensor(self.labels[index])

    def __len__(self):
        return len(self.features)


class Net(nn.Module):
    """
    """

    def __init__(self, layer_shape, beta=1, tao=2, lambda_=1, learning_rate=0.001):
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
            layer = nn.Linear(layer_shape[_], layer_shape[_ + 1])
            self.layers.append(layer)
            self.add_module("layer{}".format(_), layer)

        self.beta = beta
        self.tao = tao
        self.lambda_ = lambda_
        self.loss = 0.0
        self.learning_rate = learning_rate
        self.gradient = []
        self.logger = logging.getLogger(__name__)

    def forward(self, features):
        """
        Do forward.
        -----------
        :param features: Variable, feature
        :return:
        """
        # self.logger.debug("Forward(). Input data shape: %s.", x.size())

        # project features through net
        for layer in self.layers:
            features = layer(features)
            features = F.tanh(features)

        return features

    def compute_gradient(self, sample1, sample2):
        """
        Compute gradient.
        -----------------
        :param sample1:
        :param sample2:
        :return:
        """

        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params = list(zip(params[::2], params[1::2]))

        l = 0

        feature1 = sample1['feature']
        feature2 = sample2['feature']

        if int(sample1['label'].data) == int(sample2['label'].data):
            l = 1
        else:
            l = -1

        # calculate zi(m) and h(m)
        # zi(m) is the output of m-th layer without function tanh(x)

        z1 = []
        h1 = [feature1]
        z2 = []
        h2 = [feature2]

        x1 = feature1
        x2 = feature2
        for m in range(self.layer_count - 1):
            layer = self.layers[m]
            x1 = layer(x1)
            x2 = layer(x2)
            z1.append(x1)
            z2.append(x2)
            x1 = F.tanh(x1)
            x2 = F.tanh(x2)
            h1.append(x1)
            h2.append(x2)

        self.logger.debug("z(m) and h(m) complete.")

        # calculate delta_ij(m)
        delta_12_m = [0 for m in range(self.layer_count - 1)]
        delta_21_m = [0 for m in range(self.layer_count - 1)]

        # calculate delta_ij(M)
        M = self.layer_count - 1 - 1

        # calculate c
        c = 1 - l * (self.tao - ((self(feature1) - self(feature2)).norm()) ** 2)

        delta_12_m[M] = self._g_derivative(c) * l * self._s_derivative(z1[M])
        delta_21_m[M] = self._g_derivative(c) * l * self._s_derivative(z2[M])

        for m in reversed(range(M)):
            delta_12_m[m] = torch.mm(delta_12_m[m + 1], params[m + 1][0]) * self._s_derivative(z1[m])
            delta_21_m[m] = torch.mm(delta_21_m[m + 1], params[m + 1][0]) * self._s_derivative(z2[m])

        self.logger.debug("delta_ij(m) complete.")

        # calculate partial derivative of W
        partial_derivative_W_m = []

        for m in range(self.layer_count - 1):
            temp = (self.lambda_ * params[m][0]) + (delta_12_m[m] * h1[m].t()).t() + (delta_21_m[m] * h2[m].t()).t()
            partial_derivative_W_m.append(temp)

        self.logger.debug("partial_derivative_W(m) complete.")

        # calculate partial derivative of b
        partial_derivative_b_m = []

        for m in range(self.layer_count - 1):
            temp = self.lambda_ * params[m][1] + delta_12_m[m] + delta_21_m[m]
            partial_derivative_b_m.append(temp)

        self.logger.debug("partial_derivative_b(m) complete.")

        # combine two partial derivatve vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        self.gradient = gradient

    def backward(self):
        """
        """

        if self.gradient:
            # update parameters
            for i, param in enumerate(self.parameters()):
                param.data = param.data.sub(self.learning_rate * self.gradient[i].data)

            # clear gradient
            del self.gradient[:]
        else:
            self.logger.info("Gradient is not computed.")

    def compute_loss(self, sample1, sample2):

        loss2 = 0.0

        feature1 = sample1['feature']
        feature2 = sample2['feature']

        if int(sample1['label'].data) == int(sample2['label'].data):
            l = 1
        else:
            l = -1

        # J1
        c = 1 - l * (self.tao - ((self(feature1) - self(feature2)).norm()) ** 2)
        loss1 = self._g(c) / 2

        self.logger.debug("J1 = %f", loss1)

        # J2
        for _ in list(self.parameters()):
            loss2 += _.norm()

        loss2 = self.lambda_ * loss2 / 2

        self.logger.debug("J2 = %f", loss2)

        self.loss = float(loss1 + loss2)

    def is_similar(self, feature1, feature2):
        distance = ((self(feature1) - self(feature2)).norm()) ** 2

        if distance <= self.tao - 1:
            return True
        else:
            return False

    def _g(self, z):
        """
        Generalized logistic loss function.
        -----------------------------------
        :param z:
        """
        return float((np.log(1 + np.exp(self.beta * z))) / self.beta)

    def _g_derivative(self, z):
        """
        The derivative of g(z).
        -----------------------
        :param z:
        """
        return float(1 / (np.exp(-1 * self.beta * z) + 1))

    @staticmethod
    def _s_derivative(z):
        """
        The derivative of tanh(z).
        --------------------------
        :param z:
        """
        return 1 - F.tanh(z) ** 2


if __name__ == '__main__':
    max_iter_count = 1
    layer_shape = (784, 392, 28, 10)

    logger = setup_logger(level=logging.INFO)

    data = DDMLDataset()
    data_loader = DataLoader(dataset=data)
    net = Net(layer_shape)
    # net.cuda()

    output = []

    for i, (image, label) in enumerate(data_loader):
        image = Variable(image)
        label = Variable(label)
        output.append({"feature": image, "label": label})
        if i % 2 == 1:
            net.compute_gradient(output[0], output[1])
            net.backward()
            net.compute_loss(output[0], output[1])
            logger.info("Iteration: %d, Loss:%f", (i + 1) // 2, net.loss)
            del output[:]



import os
import random
import logging
import math
import torch
import torch.cuda as cuda
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.
    -------------
    :param level:
    :return: logger
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

    def __init__(self, size=10, label=None):
        """
        :param label: the label must in the pairs.
        :param size: size of the dataloader.
        """
        self.data = []

        if cuda.is_available():
            tensor = cuda.FloatTensor
        else:
            tensor = FloatTensor

        if label is not None:
            if label in DDMLDataset.labels:

                self.label = label

                while len(self.data) < size:
                    while True:
                        s1 = random.choice(DDMLDataset.dataset)
                        s2 = random.choice(DDMLDataset.dataset)
                        if int(s1[1][0]) == self.label:
                            break

                    self.data.append(((tensor(s1[0]) / 255, tensor(s1[1])), (tensor(s2[0]) / 255, tensor(s2[1]))))
            else:
                raise ValueError("Label not in the dataset.")

        else:
            while len(self.data) < size:
                s1 = random.choice(DDMLDataset.dataset)
                s2 = random.choice(DDMLDataset.dataset)
                self.data.append(((tensor(s1[0]) / 255, tensor(s1[1])), (tensor(s2[0]) / 255, tensor(s2[1]))))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DDMLNet(nn.Module):

    def __init__(self, layer_shape, beta=1.0, tao=5.0, b=1.0, learning_rate=0.001):
        """

        :param layer_shape:
        :param beta:
        :param tao:
        :param learning_rate:
        """
        super(DDMLNet, self).__init__()

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
        self.softmax = nn.Softmax(dim=1)

        self._s = F.tanh

        self.beta = beta
        self.tao = tao
        self.b = b
        self.learning_rate = learning_rate
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

    def forward(self, inputs):
        """
        Do forward.
        -----------
        :param inputs: Variable, feature
        :return:
        """

        # project features through net
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self._s(x)

        return x

    def softmax_forward(self, inputs):
        """
        Do softmax forward.
        -------------------
        :param inputs: Variable, feature
        :return:
        """

        x = self(inputs)
        x = self.softmax_layer(x)
        x = self._s(x)
        x = self.softmax(x)

        return x

    def loss(self, dataloader):
        """
        Compute loss.
        -------------
        :param dataloader: DataLoader of train data batch.
        :return:
        """

        loss = 0.0

        for si, sj in dataloader:
            xi = Variable(si[0])
            xj = Variable(sj[0])
            yi = si[1]
            yj = sj[1]

            if int(yi) == int(yj):
                l = 1
            else:
                l = -1

            dist = self.compute_distance(xi, xj)
            c = self.b - l * (self.tao - dist)
            loss += self._g(c)

        loss = loss / 2

        self.logger.debug("Loss = %f", loss)

        return loss

    def _softmax_backward(self, dataloader):

        cel = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        for index, (si, sj) in enumerate(dataloader):
            xi = Variable(si[0], requires_grad=True)
            xj = Variable(sj[0], requires_grad=True)

            if cuda.is_available():
                yi = Variable(si[1].type(cuda.LongTensor).squeeze())
                yj = Variable(sj[1].type(cuda.LongTensor).squeeze())
            else:
                yi = Variable(si[1].type(LongTensor).squeeze())
                yj = Variable(sj[1].type(LongTensor).squeeze())

            optimizer.zero_grad()
            xi = self.softmax_forward(xi)
            loss_i = cel(xi, yi)
            loss_i.backward()
            optimizer.step()

            optimizer.zero_grad()
            xj = self.softmax_forward(xj)
            loss_j = cel(xj, yj)
            loss_j.backward()
            optimizer.step()

    def _compute_gradient(self, dataloader):
        """

        :param dataloader:
        :return: gradient.
        """
        # W lies in 0, 2, 4...
        # b lies in 1, 3, 5...
        params = list(self.parameters())
        params_W = params[::2]
        params_b = params[1::2]

        # calculate z(m) and h(m)
        # z(m) is the output of m-th layer without function tanh(x)
        z_i_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        h_i_m = [[0 for m in range(self.layer_count)] for index in range(len(dataloader))]
        z_j_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        h_j_m = [[0 for m in range(self.layer_count)] for index in range(len(dataloader))]

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

        delta_ij_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]
        delta_ji_m = [[0 for m in range(self.layer_count - 1)] for index in range(len(dataloader))]

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

            c = self.b - l * (self.tao - self.compute_distance(xi, xj))

            # h(m) have M + 1 values and m start from 0, in fact, delta_ij_m have M value and m start from 1
            delta_ij_m[index][M] = (self._g_derivative(c) * l * (h_i_m[index][M + 1] - h_j_m[index][M + 1])) * self._s_derivative(z_i_m[index][M])
            delta_ji_m[index][M] = (self._g_derivative(c) * l * (h_j_m[index][M + 1] - h_i_m[index][M + 1])) * self._s_derivative(z_j_m[index][M])

        # calculate delta(m)

        for index in range(len(dataloader)):
            for m in reversed(range(M)):
                delta_ij_m[index][m] = torch.mm(delta_ij_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_i_m[index][m])
                delta_ji_m[index][m] = torch.mm(delta_ji_m[index][m + 1], params_W[m + 1]) * self._s_derivative(z_j_m[index][m])

        # calculate partial derivative of W
        partial_derivative_W_m = [0 * params_W[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_W_m[m] += (delta_ij_m[index][m] * h_i_m[index][m].t()).t() + (delta_ji_m[index][m] * h_i_m[index][m].t()).t()

        # calculate partial derivative of b
        partial_derivative_b_m = [0 * params_b[m] for m in range(self.layer_count - 1)]

        for m in range(self.layer_count - 1):
            for index in range(len(dataloader)):
                partial_derivative_b_m[m] += (delta_ij_m[index][m] + delta_ji_m[index][m]).squeeze()

        # combine two partial derivative vectors
        gradient = []

        for m in range(self.layer_count - 1):
            gradient.append(partial_derivative_W_m[m])
            gradient.append(partial_derivative_b_m[m])

        return gradient

    def backward(self, dataloader):
        """
        Doing backward propagation.
        ---------------------------
        :param dataloader: DataLoader of train data pairs batch.
        """
        #
        # softmax backward
        #
        self._softmax_backward(dataloader)

        #
        # pairwise backward
        #
        # gradient = self._compute_gradient(dataloader)

        # update parameters
        for i, param in enumerate(self.parameters()):
            if i < 2 * (self.layer_count - 1):
                param.data.sub_(self.learning_rate * gradient[i].data)

    def compute_distance(self, input1, input2):
        """
        Compute the distance of two samples.
        ------------------------------------
        :param input1: Variable
        :param input2: Variable
        :return: The distance of the two sample.
        """
        return (self(input1) - self(input2)).data.norm() ** 2


def main():
    test_label = 0

    train_epoch_number = 1000
    train_batch_size = 1
    test_data_size = 10000

    layer_shape = (784, 1568, 196)

    # logger = setup_logger()
    logger = setup_logger(level=logging.INFO)

    test_data = DDMLDataset(label=test_label, size=test_data_size)
    test_data_loader = DataLoader(dataset=test_data)

    net = DDMLNet(layer_shape, beta=1.0, tao=10.0, b=1.0, learning_rate=0.001)

    pkl = "pkl/ddml({}: {}-{}-{}).pkl".format(test_label, layer_shape, net.beta, net.tao)
    txt = "pkl/ddml({}: {}-{}-{}).txt".format(test_label, layer_shape, net.beta, net.tao)

    if cuda.is_available():
        net.cuda()
        logger.info("Using cuda!")

    if os.path.exists(pkl):
        state_dict = torch.load(pkl)
        net.load_state_dict(state_dict)
        logger.info("Load state from file.")

    loss_sum = 0.0

    for epoch in range(train_epoch_number):
        train_data = DDMLDataset(label=None, size=train_batch_size)
        train_data_loader = DataLoader(dataset=train_data)
        net.backward(train_data_loader)
        loss = net.loss(train_data_loader)
        loss_sum += loss
        logger.info("Iteration: %6d, Loss: %6.3f, Average Loss: %6.3f", epoch + 1, loss, loss_sum / (epoch + 1))

    torch.save(net.state_dict(), pkl)

    #
    # test (with test label)
    #

    similar_dist_sum = 0.0
    dissimilar_dist_sum = 0.0
    similar_incorrect = 0
    dissimilar_incorrect = 0
    similar_correct = 0
    dissimilar_correct = 0
    num = 0

    distance_list = [0 for l in DDMLDataset.labels]
    pairs_count = [0 for l in DDMLDataset.labels]

    for si, sj in test_data_loader:
        xi = Variable(si[0])
        yi = int(si[1])
        xj = Variable(sj[0])
        yj = int(sj[1])

        actual = (yi == yj)
        dist = net.compute_distance(xi, xj)
        result = (dist <= net.tao)

        distance_list[yj] += dist
        pairs_count[yj] += 1

        if actual:
            similar_dist_sum += dist
            if result:
                similar_correct += 1
            else:
                similar_incorrect += 1
        else:
            dissimilar_dist_sum += dist
            if not result:
                dissimilar_correct += 1
            else:
                dissimilar_incorrect += 1

        num += 1

        logger.info("%6d, %2d, %2d, %9.3f", num, int(yi), int(yj), dist)

    logger.info("Similar: Average Distance: %.6f", similar_dist_sum / (similar_correct + similar_incorrect))
    logger.info("Dissimilar: Average Distance: %.6f", dissimilar_dist_sum / (dissimilar_correct + dissimilar_incorrect))
    logger.info("\nConfusion Matrix:\n\t%6d\t%6d\n\t%6d\t%6d", similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct)

    with open(txt, mode='a') as t:

        print('Average Loss: {:6.3f}'.format(loss_sum / train_epoch_number), file=t)
        print("Confusion Matrix:\n\t{:6d}\t{:6d}\n\t{:6d}\t{:6d}".format(similar_correct, similar_incorrect, dissimilar_incorrect, dissimilar_correct), file=t)

        print('   ', end='', file=t)
        for label in DDMLDataset.labels:
            print('{:^7}'.format(label), end='\t', file=t)
        print('\n{}: '.format(test_label), end='', file=t)

        for l in DDMLDataset.labels:
            try:
                v = '{:.3f}'.format(distance_list[l] / pairs_count[l])
            except ZeroDivisionError:
                v = ' None '

            print(v, end='\t', file=t)

        print('\n', file=t)


if __name__ == '__main__':
    main()
    # test_label = 0
    #
    # train_epoch_number = 1000
    # train_batch_size = 10
    # test_data_size = 1
    #
    # layer_shape = (784, 1568, 196)
    #
    # net = DDMLNet(layer_shape, beta=2.5, tao=20.0, b=5.0, learning_rate=0.001)
    #
    # test_data = DDMLDataset(label=test_label, size=test_data_size)
    # test_data_loader = DataLoader(dataset=test_data)
    #
    # for epoch in range(train_epoch_number):
    #     train_data = DDMLDataset(label=test_label, size=train_batch_size)
    #     train_data_loader = DataLoader(dataset=train_data)
    #     net._softmax_backward(train_data_loader)

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNN(nn.Module):
    def __init__(self, n_in, h_1, h_2):
        """
        constructor of MyNN
        :param n_in: input dimension
        :param h_1: number of neurons in the first hidden layer
        :param h_2: number of neurons in the second hidden layer
        """
        super(MyNN, self).__init__()
        self.linear1 = nn.Linear(n_in, h_1)  # input and first hidden layer
        self.linear2 = nn.Linear(h_1, h_2)  # two hidden layers
        self.linear3 = nn.Linear(h_2, 1)  # second hidden layer and output layer

    def forward(self, X):
        """
        forward pass
        :param X: input feature vector
        :return: a vector of confidence score
        """
        X = self.linear1(X)
        X = F.relu(X)  # first activation function: relu
        X = self.linear2(X)
        X = torch.relu(X)  # second activation function: relu
        X = self.linear3(X)
        X = torch.sigmoid(X)  # sigmoid function here

        return X

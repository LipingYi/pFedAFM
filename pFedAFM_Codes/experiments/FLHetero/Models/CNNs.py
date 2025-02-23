from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math
import torch


class CNN_1_homo_AFM(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_homo_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o = F.relu(self.fc2(x))
        x = self.fc3(o)
        return x, o



class CNN_1_hetero_AFM(nn.Module):
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_1_hetero_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)


    def forward(self, x, homo_rep, alpha):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o1 = F.relu(self.fc2(x))
        o = o1*alpha.to(homo_rep.device) + homo_rep
        x = self.fc3(o)
        return x, o

class CNN_2_hetero_AFM(nn.Module): # change filters of convs
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_2_hetero_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels, 5)
        self.fc1 = nn.Linear(n_kernels * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o1 = F.relu(self.fc2(x))
        o = o1*alpha.to(homo_rep.device) + homo_rep
        x = self.fc3(o)
        return x, o

class CNN_3_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_3_hetero_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o1 = F.relu(self.fc2(x))
        o = o1*alpha.to(homo_rep.device) + homo_rep
        x = self.fc3(o)
        return x, o


class CNN_4_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_4_hetero_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o1 = F.relu(self.fc2(x))
        o = o1*alpha.to(homo_rep.device) + homo_rep
        x = self.fc3(o)
        return x, o

class CNN_5_hetero_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_hetero_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x, homo_rep, alpha):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o1 = F.relu(self.fc2(x))
        o = o1*alpha.to(homo_rep.device) + homo_rep
        x = self.fc3(o)
        return x, o

class CNN_5_homo_AFM(nn.Module): # change dim of FC
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10):
        super(CNN_5_homo_AFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2* n_kernels, 5)
        self.fc1 = nn.Linear(2* n_kernels * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        o = F.relu(self.fc2(x))
        x = self.fc3(o)
        return x, o

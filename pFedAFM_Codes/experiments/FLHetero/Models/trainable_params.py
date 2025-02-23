from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
import numpy as np
import math
import torch


class vector_alpha(nn.Module):
    def __init__(self):
        super(vector_alpha, self).__init__()

        self.alpha = nn.Parameter(torch.ones(500))

    def forward(self, small_input):
        output = (1-self.alpha.to(small_input.device)) * small_input
        return output



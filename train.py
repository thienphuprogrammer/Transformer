import math
import time

import pylab as p
from torch import nn, optim
from torch.optim import Adam

from models.model.transformer import Transformer
from data import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model = Transformer(s)
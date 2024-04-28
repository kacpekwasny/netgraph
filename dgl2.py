import itertools

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dlg.data 

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]


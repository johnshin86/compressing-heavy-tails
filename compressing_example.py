"""
This shows an example run used to create the entries of Table 1 in the paper. 
Due to the random nature of the compression, the compressed accuracy will 
likely not exactly match the paper. 
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pytorchcv
from pytorchcv.model_provider import get_model as ptcv_get_model

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

from compressing_ht import *

_,_, train_loader, test_loader = get_data_cifar10(train_batch_size=100, test_batch_size=100, random_labels=False)

thismodel = 'densenet40_k12_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)

data = model.output.weight.detach().numpy().flatten()

a = fit_pl(model.output.weight.detach())

pre, post = compress_dense_compare(model, test_loader, iterations=10)


print(a)
print(pre)
print(post)
print(np.mean(post))
print(np.std(post))
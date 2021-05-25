"""
This shows an example run used to create the entries of Table 1 in the paper. 
Due to the random nature of the compression, the compressed accuracy will 
likely not exactly match the paper. 
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import pytorchcv
import powerlaw
from pytorchcv.model_provider import get_model as ptcv_get_model

from compressing_ht import *

_,_, train_loader, test_loader = get_data_cifar10(train_batch_size=100, test_batch_size=100, random_labels=False)

thismodel = 'densenet40_k12_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model.to("cuda") #to make the compress_dense_compare function work faster

data = model.output.weight.detach().cpu().numpy().flatten()

a = fit_pl(model.output.weight.detach().cpu())

pre, post = compress_dense_compare(model, test_loader, iterations=10)


print(a)
print(pre)
print(post)
print(np.mean(post))
print(np.std(post))
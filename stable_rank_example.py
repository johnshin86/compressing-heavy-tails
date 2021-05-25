"""
This notebook shows how to create the stable rank plot in the paper. 
Due to the random nature of EM, the linear fit may not match the paper. 
"""

import numpy as np
import matplotlib.pyplot as pyplot
import seaborn as sns
import scipy as sp 
import scipy.linalg as lin

import torch
import pytorchcv
from pytorchcv.model_provider import get_model as ptcv_get_model
import powerlaw

from compressing_ht import *

alphas = []
srs = []

thismodel = 'resnet20_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())

thismodel = 'resnet56_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet110_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet164bn_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet272bn_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet20_cifar100'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet56_cifar100'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'resnet110_cifar100'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'densenet40_k12_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'densenet40_k12_cifar100'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'densenet40_k12_bc_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'xdensenet40_2_k24_bc_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'xdensenet40_2_k24_bc_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'resnet20_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'resnet56_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'resnet110_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'pyramidnet110_a48_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'pyramidnet110_a48_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'pyramidnet110_a84_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet20_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet56_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet110_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet20_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet56_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'preresnet110_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'seresnet20_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

thismodel = 'seresnet56_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'seresnet110_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'seresnet20_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'seresnet56_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'seresnet110_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet20_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet56_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet110_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet20_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet56_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'sepreresnet110_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn16_10_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn16_10_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn28_10_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn28_10_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn40_8_cifar10'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model


thismodel = 'wrn40_8_svhn'
model = ptcv_get_model(thismodel, pretrained=True)
model = model.to("cuda")
sr = stable_rank(list(model.children())[-1].weight.detach())
a = fit_pl(list(model.children())[-1].weight.detach().cpu())
alphas.append(a)
srs.append(sr.cpu())
del model

sns.set_theme()

x = np.zeros((len(alphas),2))
x[:,0] = alphas
x[:,1] = 1.
y = srs

r = np.linspace(np.min(alphas), np.max(alphas), 100)

rx=np.ones( (len(alphas),2) )
r=np.arange(np.min(alphas),np.max(alphas), (np.max(alphas)-np.min(alphas))/len(alphas))
rx[:,0]=r

rw1,rw2=em(x, y)
plt.plot(r,np.dot(rx,rw1),'-r')
plt.plot(r,np.dot(rx,rw2),'-g')

plt.scatter(alphas, srs)
plt.ylabel(r"Stable rank")
plt.xlabel(r"$\alpha$-fit")
#plt.title(r"Stable Rank as a Function of $\alpha$")
plt.savefig('stable_rank.svg', format='svg', dpi=1200)

plt.show()
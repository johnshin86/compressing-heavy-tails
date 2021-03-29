import powerlaw

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision import transforms

import math
import numpy as np
import scipy as sp 
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg as lin 


def power_method(W, iterations=100, device="cuda:0"):
	"""Computes the top eigenvalue of a matrix.

	Input: the matrix, M
	Optional: iterations (default: 100), device (default: cuda:0)
	Return: the largest eigenvalue of M.
	"""
	_ , m = W.shape

	M = W.t() @ W
	vk = torch.empty(m, device=device).normal_(mean=0, std=1.)

	for i in range(iterations):
		vk1 = M @ vk
		vk1_norm = torch.norm(vk1)
		vk = vk1 / vk1_norm

	top_eig = vk @ M @ vk
	del vk
	del vk1

	return top_eig

def stable_rank(M, device="cuda:0"):
	"""Computes the stable rank of a matrix.
	Input: the matrix, M
	Return: Stable rank of M, ||M||^2_F / ||M||^2_2
	"""
	if device == "cuda:0":
		M.to("cuda:0")
	frob = torch.linalg.norm(M, ord='fro')**2
	spectral = power_method(M, device = device)**2
	return frob/spectral


def generate_pl(N_1, N_2, a):
	""" Generates a matrix where the entries are given by a power-law
	Input: matrix rows (N_1), matrix columns (N_2), power-law exponent (a) 

	Return: matrix with power-law entries, M
	"""
	M = torch.zeros((N_1, N_2)).to("cuda:0")
	for i in range(N_1):
		for j in range(N_2):
			M[i][j] = np.random.pareto(a)
	return M

def fit_pl(M):
	"""Fits a power-law to the matrix elements of M.

	Input: M, matrix
	Output: power law exponent (a), R-square value (R), p-value (p)
	"""
	M = M.flatten()
	results = powerlaw.Fit(M)
	a = results.power_law.alpha
	#R, p = results.distribution_compare('power_law', 'lognormal')
	return a #, R, p

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


def compress_dense_compare(model, test_loader, device="cuda:0", iterations = 5):
	"""Computes the accuracy of a model pre- and post- compressing the final dense layer.

	Input: model, test_loader, device (default: "cuda:0")
	Output: pre_test_acc, post_test_accs
	"""
	pre_test_acc = compute_acc(model, test_loader, device="cuda:0").cpu().numpy()
	post_test_accs = []

	W = model.output.weight.detach()
	i_max, j_max = W.size()
	mean1 = torch.mean(W.flatten())
	std1 = torch.std(W.flatten())

	W_A_mask = torch.abs(W) <= std1
	W_B_mask = torch.abs(W) > std1
	W_B = torch.mul(W_B_mask, W)

	for i in range(iterations):
		W_gauss_tmp = torch.empty((i_max, j_max)).normal_(mean=mean1, std=std1).to(device)
		W_gauss = torch.mul(W_A_mask, W_gauss_tmp)
		W_new = W_gauss + W_B
		with torch.no_grad():
			list(model.children())[-1].weight = torch.nn.Parameter(W_new)
		post_test_acc = compute_acc(model, test_loader, device="cuda:0").cpu().numpy()
		post_test_accs.append(post_test_acc)
	post_test_accs = np.array(post_test_accs)
	return pre_test_acc, post_test_accs


def compute_acc(model, data_loader, device, class_label_filter=None):
	"""Compute the accuracy of a classifier given the model, as a percentage.
	Input: model, data_loader, device
	Return: accuracy (%)
	"""
	correct_pred, num_examples = 0, 0
	model.eval()
	if class_label_filter == None:
		for i, (features, targets) in enumerate(data_loader):
			features, targets = features.to(device), targets.to(device)
			probas = model(features)
			_, predicted_labels = torch.max(probas, 1)
			num_examples += targets.size(0)
			assert predicted_labels.size() == targets.size()
			correct_pred += (predicted_labels == targets).sum()
	else:
		for i, (features, targets) in enumerate(data_loader):
			indices = [i for i, x in enumerate(targets) if x == class_label_filter]
			features = features[indices].to(device)
			targets = targets[indices].to(device)
			probas = model(features)
			_, predicted_labels = torch.max(probas, 1)
			num_examples += targets.size(0)
			assert predicted_labels.size() == targets.size()
			correct_pred += (predicted_labels == targets).sum()

	return correct_pred.float()/num_examples * 100

def get_data_cifar10(
		train_batch_size=100,
		test_batch_size=100,
		train_range=None, 
		test_range=None, 
		random_labels=False, 
		seed = 0):
	"""Get CIFAR10 data. If random_labels=True, randomizes the labels. 
	Inputs: train_batch_size (default: 100), test_batch_size (default:100), train_range (default: None), test_range (default: None), random_labels (default: False), seed (default: None)
	Return: train dataset, test dataset, train loader, test loader
	"""
	normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	train_dataset = datasets.CIFAR10(
								root='data', 
								train=True, 
								transform=transform_train,
								download=True)
	test_dataset = datasets.CIFAR10(
								root='data', 
								train=False, 
								transform=transform_test,
								download=True)
	if random_labels:
		print("generating random labels with seed {}".format(seed))
		np.random.seed(seed)

		probability_of_random = 1.0
		labels = np.array(train_dataset.targets) 
		mask = np.random.rand(len(labels)) <= probability_of_random #create mask of length labels, where entries drawn from [0,1].
		rnd_labels = np.random.choice(10, mask.sum())               #create random labels 1-10 of length of mask
		labels[mask] = rnd_labels
		labels = [int(x) for x in labels]
		train_dataset.targets = labels                              #assign new random labels to dataset
		np.savetxt("random_labels.txt", labels)

	if train_range:
		train_dataset = Subset(train_dataset, train_range)

	if test_range:
		test_dataset = Subset(test_dataset, test_range)


	train_loader = DataLoader(
		dataset=train_dataset, 
		batch_size=train_batch_size,
		num_workers=4,
		shuffle=False)
	test_loader = DataLoader(
		dataset=test_dataset, 
		batch_size=test_batch_size,
		num_workers=4,
		shuffle=False)
	return train_dataset, test_dataset, train_loader, test_loader

def get_data_cifar100(
		train_batch_size=100,
		test_batch_size=100,
		train_range=None, 
		test_range=None, 
		random_labels=False, 
		seed = 0):
	"""Get CIFAR100 data. If random_labels=True, randomizes the labels. 
	Inputs: train_batch_size (default: 100), test_batch_size (default:100), train_range (default: None), test_range (default: None), random_labels (default: False), seed (default: None)
	Return: train dataset, test dataset, train loader, test loader
	"""
	normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	train_dataset = datasets.CIFAR100(
								root='data', 
								train=True, 
								transform=transform_train,
								download=True)
	test_dataset = datasets.CIFAR100(
								root='data', 
								train=False, 
								transform=transform_test,
								download=True)
	if random_labels:
		print("generating random labels with seed {}".format(seed))
		np.random.seed(seed)

		probability_of_random = 1.0
		labels = np.array(train_dataset.targets) 
		mask = np.random.rand(len(labels)) <= probability_of_random #create mask of length labels, where entries drawn from [0,1].
		rnd_labels = np.random.choice(10, mask.sum())               #create random labels 1-10 of length of mask
		labels[mask] = rnd_labels
		labels = [int(x) for x in labels]
		train_dataset.targets = labels                              #assign new random labels to dataset
		np.savetxt("random_labels.txt", labels)

	if train_range:
		train_dataset = Subset(train_dataset, train_range)

	if test_range:
		test_dataset = Subset(test_dataset, test_range)


	train_loader = DataLoader(
		dataset=train_dataset, 
		batch_size=train_batch_size,
		num_workers=4,
		shuffle=False)
	test_loader = DataLoader(
		dataset=test_dataset, 
		batch_size=test_batch_size,
		num_workers=4,
		shuffle=False)
	return train_dataset, test_dataset, train_loader, test_loader

def get_data_svhn(
		train_batch_size=100,
		test_batch_size=100,
		train_range=None, 
		test_range=None, 
		random_labels=False, 
		seed = 0):
	"""Get SVHN data. If random_labels=True, randomizes the labels. 
	Inputs: train_batch_size (default: 100), test_batch_size (default:100), train_range (default: None), test_range (default: None), random_labels (default: False), seed (default: None)
	Return: train dataset, test dataset, train loader, test loader
	"""
	normalize = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442], std= [0.19803012, 0.20101562, 0.19703614])
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	transform_test = transforms.Compose([
		transforms.ToTensor(),
		normalize])
	train_dataset = datasets.SVHN(
								root='data', 
								split='train', 
								transform=transform_train,
								download=True)
	test_dataset = datasets.SVHN(
								root='data', 
								split='test', 
								transform=transform_test,
								download=True)
	if random_labels:
		print("generating random labels with seed {}".format(seed))
		np.random.seed(seed)

		probability_of_random = 1.0
		labels = np.array(train_dataset.targets) 
		mask = np.random.rand(len(labels)) <= probability_of_random #create mask of length labels, where entries drawn from [0,1].
		rnd_labels = np.random.choice(10, mask.sum())               #create random labels 1-10 of length of mask
		labels[mask] = rnd_labels
		labels = [int(x) for x in labels]
		train_dataset.targets = labels                              #assign new random labels to dataset
		np.savetxt("random_labels.txt", labels)

	if train_range:
		train_dataset = Subset(train_dataset, train_range)

	if test_range:
		test_dataset = Subset(test_dataset, test_range)


	train_loader = DataLoader(
		dataset=train_dataset, 
		batch_size=train_batch_size,
		num_workers=4,
		shuffle=False)
	test_loader = DataLoader(
		dataset=test_dataset, 
		batch_size=test_batch_size,
		num_workers=4,
		shuffle=False)
	return train_dataset, test_dataset, train_loader, test_loader

def em(lam=.01, eta=100, x, y):
	rpi = np.zeros((2)) + .5
	pi = np.zeros((len(x), 2)) + 0.5 
	w1 = np.random.rand(2)
	w2 = np.random.rand(2)

	for _ in range(100):
		plt.plot(r, np.dot(rx, w1), '-r', alpha=0.5)
		plt.plot(r, np.dot(rx, w2), '-g', alpha=0.5)

	err1 = y - np.dot(x, w1)
	err2 = y - np.dot(x, w2)

	prbs = np.zeros( (len(y), 2))
	prbs[:,0] = -.5*eta*err1**2
	prbs[:,1] = -.5*eta*err2**2

	pi = np.tile(rpi, (len(x), 1))*np.exp(prbs)
	pi /= np.tile(np.sum(pi, 1), (2,1)).T

	rpi = np.sum(pi, 0)
	rpi /= np.sum(rpi)

	pi1x = np.tile(pi[:,0], (2,1)).T*x
	xp1 = np.dot(pi1x.T, x) + np.eye(2)*lam/eta
	yp1 = np.dot(pi1x.T, y)
	w1 = lin.solve(xp1, yp1)

	pi2x=np.tile(pi[:,1],(2,1)).T*x
    xp2=np.dot(pi2x.T,x)+np.eye(2)*lam/eta
    yp2=np.dot(pi[:,1]*y,x)
    w2=lin.solve(xp2,yp2)

    eta = np.sum(pi)/np.sum(-prbs/eta*pi)

    obj=np.sum(prbs*pi)-np.sum(pi[pi>1e-50]*np.log(pi[pi>1e-50]))+np.sum(pi*np.log(np.tile(rpi,(len(x),1))))+np.log(eta)*np.sum(pi)
    print(obj,eta,rpi,w1,w2)

    try:
    	if np.isnan(obj): break
    	if np.abs(obj-oldobj)<1e-2: break
    except:
    	pass

    oldobj=obj

    return w1, w2 
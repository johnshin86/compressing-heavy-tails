import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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

def get_data(
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
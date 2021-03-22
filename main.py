import numpy as np 
import torch
import scipy as sp 
import powerlaw



def power_method(M, iterations=100, device="cuda:0"):
	"""Computes the top eigenvalue of a matrix.

	Input: the matrix, M
	Optional: iterations (default: 100), device (default: cuda:0)
	Return: the largest eigenvalue of M.
	"""
	_ , m = M.shape
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
	Return: ||M||^2_F / ||M||^2_2
	"""
	if device == "cuda:0":
		M.to("cuda:0")
	frob = torch.linalg.norm(M, ord='fro')**2
	spectral = power_method(M, device = device)**2
	return frob/spectral


def generate_pl(N_1, N_2, a, loc = 0., scale=1.0):
	""" Generates a matrix where the entries are given by a power-law
	Input: matrix rows (N_1), matrix columns (N_2), power-law exponent (a) 
	location (loc, default: 0), scale (default: 0)

	Return: M, matrix with power-law entries.
	"""
	M = torch.zeros((N_1, N_2))
	for i in range(N_1):
		for j in range(N_2):
			M[i][j] = sp.stats.powerlaw.rvs(a, loc= loc, scale=scale)
	return M

def fit_pl(M):
	"""Fits a power-law to the matrix elements of M.

	Input: M, matrix
	Output: power law exponent (a), R-square value (R), p-value (p)
	"""
	M = M.flatten()
	results = powerlaw.Fit(M)
	a = results.power_law.alpha
	R, p = results.distribution_compare('power_law', 'lognormal')
	return a, R, p

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

def compress_dense_compare(model, test_loader, device="cuda:0"):
	test_acc = compute_acc(model, test_loader, device="cuda:0")
	return test_acc

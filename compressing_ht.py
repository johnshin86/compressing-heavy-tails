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
	Return: Stable rank of M, ||M||^2_F / ||M||^2_2
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

	Return: matrix with power-law entries, M
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

def rename_attribute(obj, old_name, new_name):
    obj._modules[new_name] = obj._modules.pop(old_name)


def compress_dense_compare(model, test_loader, device="cuda:0"):
	"""Computes the accuracy of a model pre- and post- compressing the final dense layer.

	Input: model, test_loader, device (default: "cuda:0")
	Output: pre_test_acc, post_test_acc
	"""
	pre_test_acc = compute_acc(model, test_loader, device="cuda:0")

	mean1 = torch.mean(model.output.weight.flatten().detach())
	std1 = torch.std(model.output.weight.flatten().detach())

	i_max, j_max = model.output.weight.size()

	for i in range(i_max):
  		for j in range(j_max):
    		if torch.abs(model.output.weight[i,j]) < std1:
      			with torch.no_grad():
        			list(model.children())[-1].weight[i][j] = torch.normal(mean=mean1, std=std1, size=(1,))

    post_test_acc = compute_acc(model, test_loader, device="cuda:0")

	return pre_test_acc, post_test_acc

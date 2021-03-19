import numpy as np 
import torch
import scipy as sp 


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
	if device == "cuda:0":
		M.to("cuda:0")
	frob = torch.linalg.norm(M, ord='fro')**2
	spectral = power_method(M, device = device)**2
	return frob/spectral


def generate_pl(N_1, N_2, a, loc = 0., scale=1.0):
	M = torch.zeros((N_1, N_2))
	for i in range(N_1):
		for j in range(N_2):
			M[i][j] = sp.stats.powerlaw.rvs(a, loc= loc, scale=scale)
	return M



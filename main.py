import numpy as np 
import torch


def stable_rank(M, cuda=True):
	if cuda == True:
		M.to("cuda")
	
	return 0.
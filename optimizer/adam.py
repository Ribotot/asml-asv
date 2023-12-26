#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised Adam optimizer')

	return torch.optim.Adam(parameters, lr = lr, betas=(0.9, 0.999), eps=1e-06, weight_decay = weight_decay, amsgrad=False);

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, max_epoch, lr_decay, decay_interval=10, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import json

def Scheduler(optimizer, decay_epochs, max_epoch, lr_decay, **kwargs):

	decay_epochs = json.loads(decay_epochs)
	sche_fn = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised Multi-step LR scheduler')

	return sche_fn, lr_step

#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import json

def Scheduler(optimizer, max_epoch, lr_decay, decay_epochs='[40, 45, 50, 55, 60, 65, 70, 75]', **kwargs):

	decay_epochs = json.loads(decay_epochs)
	sche_fn = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised Multi-step LR scheduler')

	return sche_fn, lr_step

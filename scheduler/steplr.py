#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

<<<<<<< HEAD
def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=test_interval, gamma=lr_decay)
=======
def Scheduler(optimizer, max_epoch, lr_decay, decay_interval=10, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=lr_decay)
>>>>>>> 463ada6aeb053540ce2428831b625449a57c7a09

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step

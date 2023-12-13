#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, max_epoch, lr, warmup_epoch, **kwargs):

	pct_start = float(warmup_epoch)/float(max_epoch)
	sche_fn = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=pct_start, anneal_strategy='cos')
	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step

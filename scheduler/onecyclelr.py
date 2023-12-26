#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import math

def Scheduler(optimizer, batch_size, max_epoch, lr, warmup_epoch=10, epoch_per_sample=1092009, **kwargs):

	total_iteration = math.ceil(epoch_per_sample/batch_size)*max_epoch

	pct_start = float(warmup_epoch)/float(max_epoch)
	sche_fn = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_iteration, pct_start=pct_start, anneal_strategy='cos')
	lr_step = 'iteration'

	print('Initialised OneCycle LR scheduler')

	return sche_fn, lr_step

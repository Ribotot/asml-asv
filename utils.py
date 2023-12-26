#! /usr/bin/python
# -*- encoding: utf-8 -*-

import json
import numpy as np
import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def dict2scp(scp, dicts):
    """write dictionary to scp file"""
    with open(scp, 'w') as file:
        file.write(json.dumps(dicts))

def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


## ===== ===== ===== ===== 
##  File-IO functions  
## ===== ===== ===== ===== 

def load_dict_txt(filepath, dtype=None):
    dic = dict()
    for line in open(filepath):
        key, val = line.rstrip().split(' ', 1)
        if dtype is not None:
            val = dtype(val)
        dic[key] = val
    return dic

def save_dict_npy(filepath, dictionary):
    np.save(filepath, dictionary)

def load_dict_npy(filepath):
    return np.load(filepath, allow_pickle=True).item()

def numpy_normalize(x, p=2, dim=1):
    l2 = np.atleast_1d(np.linalg.norm(x, p, dim))
    l2[l2==0] = 1
    return x / np.expand_dims(l2, dim)
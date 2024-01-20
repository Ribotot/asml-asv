#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, center=3, topk=5, submargin=0.06, sch_epoch=20, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        init_m = 0.2
        init_sub_m = 0.06
        self.m = nn.Parameter(torch.tensor(init_m), requires_grad=False)
        self.sub_m = nn.Parameter(torch.tensor(-init_sub_m), requires_grad=False)

        self.final_m = margin
        self.final_sub_m = -submargin

        self.epoch_per_m = (margin-init_m)/sch_epoch
        self.epoch_per_sub_m = (init_sub_m-submargin)/sch_epoch

        self.s = scale
        self.topk = topk
        self.nOut = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(center, nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.calculate_cos()

        print('Initialised %d Subcenter AAMSoftmax margin %.3f scale %.3f with %d topk and %.3f submargin '%(center, self.m, self.s, self.topk, -self.sub_m))
    
    def calculate_cos(self):
        
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.sub_cos_m = math.cos(self.sub_m)
        self.sub_sin_m = math.sin(self.sub_m)

        self.sub_th = math.cos(math.pi - self.sub_m)
        self.sub_mm = math.sin(math.pi - self.sub_m) * self.sub_m        

    def step(self):

        self.m += self.epoch_per_m

        if self.m > self.final_m:
            nn.init.constant_(self.m, self.final_m)
            print('\nRemain margin to %.4f'%(self.m))
        else:
            print('\nUpdate margin to %.4f'%(self.m))

        self.sub_m += self.epoch_per_sub_m

        if self.sub_m < self.final_sub_m:
            nn.init.constant_(self.sub_m, self.final_sub_m)
            print('Remain submargin to %.4f'%(-self.sub_m))
        else:
            print('Update submargin to %.4f'%(-self.sub_m))

        self.calculate_cos()

    def forward(self, x, label=None):

        batch_size = x.size()[0]
        assert batch_size == label.size()[0]
        assert x.size()[1] == self.nOut

        # cos(theta)
        cosine_subcen = torch.einsum('... j, ... a b j -> ... a b', F.normalize(x), F.normalize(self.weight, dim=-1))
        cosine, _ = torch.max(cosine_subcen, dim=1)

        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        sub_phi = cosine * self.sub_cos_m - sine * self.sub_sin_m
        sub_phi = torch.where((cosine - self.sub_th) > 0, sub_phi, cosine - self.sub_mm)

        one_hot_sub = torch.zeros_like(cosine)
        _, indecies = torch.topk(cosine, self.topk+1)

        fix_indecies = []
        for i, lab in enumerate(label):
            index = indecies[i,:]
            if lab in index:
                index = index[index != lab]
                fix_indecies.append(index.unsqueeze(dim=0))
            else:
                index = index[:-1]
                fix_indecies.append(index.unsqueeze(dim=0))
        indecies = torch.cat(fix_indecies, dim=0)

        for k in range(self.topk):
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, indecies[:,k].view(-1, 1), 1)
            one_hot_sub += one_hot

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = (one_hot * phi) + (one_hot_sub * sub_phi) +((1.0 - one_hot_sub - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(cosine.detach()*self.s, label.detach(), topk=(1,))[0]
        return loss, prec1

#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, sch_epoch=20, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        init_m = 0.0
        self.m = nn.Parameter(torch.tensor(init_m), requires_grad=False)

        self.final_m = margin
        self.epoch_per_m = (margin-init_m)/sch_epoch
        self.s = scale
        self.nOut = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin

        self.calculate_cos()

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m, self.s))
    
    def calculate_cos(self):
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def step(self):

        self.m += self.epoch_per_m

        if self.m > self.final_m:
            nn.init.constant_(self.m, self.final_m)
            print('\nRemain margin to %.4f'%(self.m))
        else:
            print('\nUpdate margin to %.4f'%(self.m))

        self.calculate_cos()

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.nOut
        
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss    = self.ce(output, label)
        prec1   = accuracy(cosine.detach()*self.s, label.detach(), topk=(1,))[0]
        return loss, prec1

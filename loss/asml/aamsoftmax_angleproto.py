#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.clova.softmax as aamsoftmax
import loss.clova.angleproto as angleproto

class LossFunction(nn.Module):

    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False, \
        init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = aamsoftmax.LossFunction(nOut=nOut, nClasses=nClasses, margin=margin, \
                                                scale=scale, easy_margin=easy_margin, **kwargs)
        self.angleproto = angleproto.LossFunction(init_w=init_w, init_b=init_b, **kwargs)

        print('Initialised AAMsoftmax & Prototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1    = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))
        nlossP1, _       = self.angleproto(x,None)
        nlossP2, _       = self.angleproto(torch.flip(x, [1]),None)
        nlossP = 0.5*(nlossP1 + nlossP2)
        

        return nlossS+nlossP, prec1

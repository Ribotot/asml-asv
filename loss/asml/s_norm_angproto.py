#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from utils import accuracy


class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, nPerSpeaker=2, init_w=10.0, init_b=-5.0, \
        margin=0.2, scale=30, cohort_size=100, easy_margin=False, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.nPerSpeaker = nPerSpeaker
        
        self.m = margin
        self.s = scale
        self.nOut = nOut
        self.nClasses = nClasses
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce1 = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

        self.cohort_size = cohort_size + 1
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

        #self.pi_2 = 3.14159274/2

        self.w2 = nn.Parameter(torch.tensor(0.0))
        self.w3 = nn.Parameter(torch.tensor(-2.8))
        self.b2 = nn.Parameter(torch.tensor(0.0))
        self.b3 = nn.Parameter(torch.tensor(2.8))
        self.ce2 = nn.CrossEntropyLoss()

        print('Initialised SN_AngleProto')

        self.weights = []
        self.weights.append(self.w)
        self.weights.append(self.b)

    def forward(self, x, label=None):
        
        batchsize = x.size()[0]
        x = x.reshape(-1,x.size()[-1])  
        label = label.repeat_interleave(2)

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

        nlossS    = self.ce(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        ## Do s_norm angproto
        x_reshape = x.reshape(batchsize, self.nPerSpeaker, self.nOut)
        cosine_reshape = cosine.clone().detach().reshape(batchsize, self.nPerSpeaker, self.nClasses)
        label_reshape = label.reshape(batchsize, self.nPerSpeaker)

        nlossP1 = self.s_norm_angproto(x_reshape, cosine_reshape, label_reshape)
        nlossP2 = self.s_norm_angproto(torch.flip(x_reshape, [1]), torch.flip(cosine_reshape, [1]), torch.flip(label_reshape, [1]))
        nlossP = 0.5*(nlossP1 + nlossP2)

        return nlossS+nlossP, prec1

    def extract_mean_std(self, x):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cohort1, _ = torch.topk(cosine_anchor, self.cohort_size, dim=-1, largest=True, sorted=True)
        var1, mean1 = torch.var_mean(cohort1, dim=-1, keepdims=True)
        std1 = torch.sqrt(var1)
        return mean1, std1

    def s_norm_angproto(self, x, cosine, label=None):

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        batchsize       = out_anchor.size()[0]

        cosine_anchor   = torch.mean(cosine[:,1:,:],1)
        cosine_positive = cosine[:,0,:]

        out_dot  = F.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))

        cohort1, index1 = torch.topk(cosine_anchor.transpose(0,1), self.cohort_size, dim=0, largest=True, sorted=True)
        cohort2, index2 = torch.topk(cosine_positive, self.cohort_size, dim=-1, largest=True, sorted=True)

        var1, mean1 = torch.var_mean(cohort1, dim=0, keepdims=True)
        std1 = torch.sqrt(var1)
        var2, mean2 = torch.var_mean(cohort2, dim=-1, keepdims=True)
        std2 = torch.sqrt(var2)

        if label != None:
            label      = label[:,0]
            else_ = 0
            coh1_list = []
            for i in range(batchsize):
                coh_ = cohort1[:,i]
                ind_ = index1[:,i]
                lab_ = label[i]
                k = (ind_ == lab_).nonzero(as_tuple=True)
                if k[0].numel() == 0:
                    coh_ = coh_[:-1]
                    else_ = 1
                else:
                    coh_ = coh_[torch.arange(self.cohort_size).to(x.device)!= k[0]]
                coh1_list.append(coh_.unsqueeze(dim=-1))
            cohort1 = torch.cat(coh1_list, dim=-1)

            coh2_list = []
            for i in range(batchsize):
                coh_ = cohort2[i,:]
                ind_ = index2[i,:]
                lab_ = label[i]
                k = (ind_ == lab_).nonzero(as_tuple=True)
                if k[0].numel() == 0:
                    coh_ = coh_[:-1]
                    else_ = 1
                else:
                    coh_ = coh_[torch.arange(self.cohort_size).to(x.device)!= k[0]]
                coh2_list.append(coh_.unsqueeze(dim=0))
            cohort2 = torch.cat(coh2_list, dim=0)
        else:
            cohort1 = cohort1[:,:-1]
            cohort2 = cohort2[:-1,:]

        if label == None:
            return mean1, mean2, std1, std2

        else:
            out_dot1 = (out_dot-F.hardsigmoid(mean1*self.w2+self.w3))/F.hardsigmoid(std1*self.b2+self.b3)
            out_dot2 = (out_dot-F.hardsigmoid(mean2*self.w2+self.w3))/F.hardsigmoid(std2*self.b2+self.b3)

            out_dot =  0.5*(out_dot1+out_dot2)
            torch.clamp(self.w, 1e-6)
            cos_sim_matrix = out_dot * self.w + self.b

            label = torch.from_numpy(numpy.asarray(range(0,batchsize))).to(x.device)
            nloss   = self.ce2(cos_sim_matrix, label)

            return nloss
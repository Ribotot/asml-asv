#! /usr/bin/python
# -*- encoding: utf-8 -*-

# revised
# Copyright 2023 Choi Jeong-Hwan

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.custom.utils import FbankAug, PreEmphasis
from models.asml.pooling import AttentiveDoubleStatsPooling


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, frequency, stride=1, downsample=None, bottleneck=128):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, 
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.GELU()
        self.se = fwSEModule(frequency, bottleneck)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class fwSEModule(nn.Module):
    def __init__(self, frequency, bottleneck):
        super(fwSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv1d(frequency, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(bottleneck, frequency, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )
    def forward(self, input):
        x1 = torch.transpose(input, -3, -2)
        x = self.avg_pool(x1)
        x = x.squeeze(dim=-1)
        x = self.se(x)
        x = x1 * x.unsqueeze(dim=-1)
        x = torch.transpose(x, -3, -2)
        return x


class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, n_mels=96, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d'%(nOut))
        
        self.inplanes   = num_filters[0]
        self.n_mels     = n_mels
        self.log_input  = log_input

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        

        self.layer1 = self._make_layer(block, num_filters[0], layers[0], n_mels)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], n_mels//2, stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], n_mels//4, stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], n_mels//8, stride=(2, 2))

        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.torchfb        = torch.nn.Sequential(
                PreEmphasis(),
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                            f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=n_mels)
                )
        self.specaug = FbankAug()

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 256, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
            )

        self.attention2 = AttentiveDoubleStatsPooling(input_dim=num_filters[3], bottle_dim=256, kernel=[1,1],  use_global_info=False,
                                                            apply_bn=True, activ="GELU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)    

        out_dim = num_filters[3] * (outmap_size * 2 + 4)
        self.fc = nn.Linear(out_dim, nOut)
        self.bn_fc = nn.BatchNorm1d(nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, frequency, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, frequency, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, frequency))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


    def _before_pooling(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _before_penultimate(self, x):
        c = self.attention2(x)

        x = x.reshape(x.size()[0],-1,x.size()[-1])

        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-6) )
        x = torch.cat((mu,sg,c),1)
        x = self.fc(x)
        x = self.bn_fc(x)
        return x

    def wave2feat(self, x, max_frame=False, aug=False):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                if max_frame:
                    audiosize = x.shape[1]
                    max_audio = max_frame*160 +240 # 16kHz defalt
                    if audiosize <= max_audio:
                        import math
                        shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
                        x = F.pad(x, (shortage,shortage), "constant", 0)          
                x = self.torchfb(x)+1e-6
                if self.log_input: x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                x = x[:,:,1:-1]
                if aug == True: x = self.specaug(x)
                x = x.unsqueeze(1).detach()
        return x


    def wave2emb(self, wave, max_frame=False, aug=False):
        feat = self.wave2feat(wave, max_frame, aug=False)
        late_feat = self._before_pooling(feat)
        emb = self._before_penultimate(late_feat)
        return emb

    def feat2emb(self, feat):
        late_feat = self._before_pooling(feat)
        emb = self._before_penultimate(late_feat)
        return emb

    def forward(self, x, max_frame=False, aug=False):
        x = self.wave2emb(x, max_frame, aug=False)
        return x

def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [48, 96, 192, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 14, 3], num_filters, nOut, **kwargs)
    return model


if __name__=="__main__":
    # batch_size, num_frames, feat_dim = 1, 3000, 80
    batch_size, second = 1, 1
    x = torch.randn(batch_size, int(second*16000))
    # x_tp = x.transpose(1, 2)

    # num_filters = [64, 128, 256, 512]
    num_filters = [48, 96, 192, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 14, 3], num_filters, nOut=256)

    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())/ 1000 / 1000
    print('Model parameters: {:.4f} M'.format(pytorch_total_params))
    # exit()
    
    torch.set_num_threads(1)
    import timeit
    model.eval()
    number = 10
    end_start = timeit.timeit(stmt='model(x)', globals=globals(), number=number)
    print('CPU Time :',end_start/number*1000, 'ms') 
    # exit()

    model.eval()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    embs = model(x)
    end.record()
    torch.cuda.synchronize()
    print('GPU Time :',start.elapsed_time(end), 'ms')
    # exit()

    from fvcore.nn import FlopCountAnalysis
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print('FLOPs: {:.4f} M'.format(flops.total()/1000/1000))
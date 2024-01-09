# Copyright 2023 Choi Jeong-Hwan

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
try:
    import models
except:
    import os, sys
    sys.path.append(os.getcwd())
    
from models.custom.utils import FbankAug, PreEmphasis

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )
    def forward(self, input):
        x = self.se(input)
        return input * x


class fwSEModule(nn.Module):
    def __init__(self, frequancy, bottleneck=128):
        super(fwSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv1d(frequancy, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, frequancy, kernel_size=1, padding=0),
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


class Global_FE(nn.Module):
    def __init__(self, inplanes, planes, outplanes, kernel_size=3, dilation=1, scale=8, bottleneck=128):
        super(Global_FE, self).__init__()

        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, planes, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(planes)
        self.conv2  = nn.Conv1d(planes, width*scale, kernel_size=1)
        self.bn2    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad, groups=width))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)

        self.se     = SEModule(planes, bottleneck)
        self.conv4  = nn.Conv1d(width*scale, outplanes, kernel_size=1)
        
        self.relu   = nn.ReLU()
        self.width  = width

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out = self.conv4(out)
        out = self.relu(out)

        return out 


class Local_FE_Block(nn.Module):
    def __init__(self, inplanes, planes, frequncy, kernel_size=(3, 3), stride=(1, 1), bottleneck=128):
        super(Local_FE_Block, self).__init__()

        self.conv1  = nn.Conv2d(inplanes, inplanes, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        # self.conv1  = nn.Conv2d(inplanes, inplanes, kernel_size=(1, 1), stride=stride)
        self.bn1    = nn.BatchNorm2d(inplanes)
        num_pad = (math.floor(kernel_size[0]/2), math.floor(kernel_size[1]/2))
        self.conv2  = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, padding=num_pad, groups=inplanes)
        self.bn2    = nn.BatchNorm2d(inplanes)
        self.conv3  = nn.Conv2d(inplanes, planes, kernel_size=(1, 1))
        self.bn3    = nn.BatchNorm2d(planes)
        self.relu   = nn.ReLU()
        self.fwse   = fwSEModule(frequncy, bottleneck)

        self.size_mismatch = False
        if stride[0] != 1 or inplanes != planes:
            self.size_mismatch = True
            self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), stride=stride)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.fwse(out)
        if self.size_mismatch:
            residual = self.conv4(residual)
        out += residual

        return out 

class ECAPA_TDNN(nn.Module):
    def __init__(self, C = 144, hidden=24, bottleneck = 128, nOut = 192, n_mels = 80, log_input = False, **kwargs):
        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=n_mels),
            )
        self.specaug = FbankAug(freq_mask_width = (0, 5), time_mask_width = (0, 5)) # Spec augmentation

        self.log_input = log_input
        self.block1D_1 = nn.Sequential(
            Local_FE_Block(1, hidden, n_mels, kernel_size=(3, 3), stride=(1, 2), bottleneck=bottleneck),
            )
        self.block1D_2 = nn.Sequential(
            Local_FE_Block(hidden, hidden, n_mels//2, kernel_size=(3, 3), stride=(2, 1), bottleneck=bottleneck),
            )
        self.block1D_3 = nn.Sequential(
            Local_FE_Block(hidden, hidden, n_mels//4, kernel_size=(3, 3),  stride=(2, 1), bottleneck=bottleneck),
            )
        self.block1D_4 = nn.Sequential(
            Local_FE_Block(hidden, hidden, n_mels//4, kernel_size=(3, 3),  stride=(1, 1), bottleneck=bottleneck),
            Local_FE_Block(hidden, hidden, n_mels//8, kernel_size=(3, 3),  stride=(2, 1), bottleneck=bottleneck),
            )
        self.block1D_5 = nn.Sequential(
            Local_FE_Block(hidden, hidden, n_mels//8, kernel_size=(3, 3),  stride=(1, 1), bottleneck=bottleneck),
            Local_FE_Block(hidden, hidden, n_mels//8, kernel_size=(3, 3),  stride=(1, 1), bottleneck=bottleneck),
            )
        self.block2D = Global_FE(hidden*n_mels//8, C, C, bottleneck=bottleneck)

        self.attention = nn.Sequential(
            nn.Conv1d(C, bottleneck, kernel_size=1),
            # nn.ReLU(),
            # nn.BatchNorm1d(bottleneck),   # I remove this layer 
            nn.Tanh(),
            nn.Conv1d(bottleneck, C, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn1 = nn.BatchNorm1d(2*C)
        self.fc2 = nn.Linear(2*C, nOut)
        self.bn2 = nn.BatchNorm1d(nOut)

    def _before_pooling(self, x):
        x = x.unsqueeze(dim=1)
        x = self.block1D_1(x)
        x = self.block1D_2(x)
        x = self.block1D_3(x)
        x = self.block1D_4(x)
        x = self.block1D_5(x)
        x = torch.flatten(x, -3, -2)
        x = self.block2D(x)
        return x

    def _before_penultimate(self, x):
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-6) )
        x = torch.cat((mu,sg),1)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
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
                x = self.torchfbank(x)+1e-6
                if self.log_input: x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                x = x[:,:,1:-1]
                if aug == True: x = self.specaug(x)
                x = x.detach()
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


def MainModel(**kwargs):
    # Number of filters
    model = ECAPA_TDNN(**kwargs)
    return model


if __name__=="__main__":
    # batch_size, num_frames, feat_dim = 1, 3000, 80
    batch_size, second = 1, 1
    x = torch.randn(batch_size, int(second*16000))
    # x_tp = x.transpose(1, 2)

    model = ECAPA_TDNN(nOut = 192)

    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())/ 1000 / 1000
    print('Model parameters: {:.4f} M'.format(pytorch_total_params))
    # exit()
    
    # torch.set_num_threads(1)
    # import timeit
    # model.eval()
    # number = 10
    # end_start = timeit.timeit(stmt='model(x)', globals=globals(), number=number)
    # print('CPU Time :',end_start/number*1000, 'ms') 
    # # exit()

    # model.eval()
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # embs = model(x)
    # end.record()
    # torch.cuda.synchronize()
    # print('GPU Time :',start.elapsed_time(end), 'ms')
    # # exit()

    from fvcore.nn import FlopCountAnalysis
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print('FLOPs: {:.4f} M'.format(flops.total()/1000/1000))
# repo:repo:lawlict/ECAPA-TDNN 
# repo:TaoRuijie/Loss-Gated-Learning 

# revised
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
            nn.GELU(),
            # nn.BatchNorm1d(bottleneck), # [TaoRuijie] remove this layer 
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, bottleneck=128):
        super(Bottle2neck, self).__init__()

        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.GELU()
        self.width  = width
        self.se     = SEModule(planes, bottleneck)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
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
        out += residual

        return out 


class ECAPA_TDNN(nn.Module):
    def __init__(self, C = 2048, bottleneck = 256, nOut = 256, n_mels = 96, log_input = True, **kwargs):
        super(ECAPA_TDNN, self).__init__()

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=n_mels),
            )
        self.specaug = FbankAug(freq_mask_width = (0, 5), time_mask_width = (0, 5)) # Spec augmentation

        self.log_input = log_input
        self.conv1  = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.GELU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8, bottleneck=bottleneck)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8, bottleneck=bottleneck)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8, bottleneck=bottleneck)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, bottleneck, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(bottleneck),
            nn.Tanh(),
            nn.Conv1d(bottleneck, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

    def _before_pooling(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)
        return x

    def _before_penultimate(self, x):
        t = x.size()[-1]
        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-6) )
        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
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
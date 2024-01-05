## Copyright 2023 Choi Jeong-Hwan 
## This is the code based on bc_res2net that we are developing for ASV task.

import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from models.custom.utils import FbankAug, PreEmphasis
from models.asml.pooling import AttentiveDoubleStatsPooling
from models.asml.bc2resnet.common import (ResNorm,
                                          Normalize,
                                          FreqAttnRes2NetUnit,
                                          Broadcast2BlockMod,
                                          Transition2BlockMod,
                                          trunc_normal_)

class FeaturePyramidSmoothingConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_sub_bands: int = 2,
                 norm_type: str = 'bn',
                 dropout: float = .1,
                 bias: bool = False):
        super(FeaturePyramidSmoothingConv, self).__init__()

        self.smoothing = Broadcast2BlockMod(
            input_dims=in_channels,
            num_sub_bands=num_sub_bands,
            norm_type=norm_type,
            bias=bias,
            dropout=dropout,
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, inputs):
        out = self.smoothing(inputs)
        out = self.pooling(out)
        return out


class FeaturePyramidModule(nn.Module):
    def __init__(self,
                 channel_list: list,
                 norm_type: str = 'bn',
                 bias: bool = False,
                 dropout: float = .1,
                 up_sample_method: str = 'bilinear',
                 down_sample_stages: list = None,
                 num_two_stage_categorization_class: int = 0,
                 **kwargs):
        super(FeaturePyramidModule, self).__init__()

        stage_ch_list = channel_list[1:]
        num_stages = len(stage_ch_list)

        self.down_ch_conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=ch,
                out_channels=stage_ch_list[0],
                kernel_size=(1, 1),
                bias=bias
            )
            for ch in reversed(stage_ch_list)
        ])

        self.up_sample_method = up_sample_method
        if self.up_sample_method == 'bilinear':
            self.up_sample = nn.Upsample(
                scale_factor=(2, 2),
                mode='bilinear',
                align_corners=True
            )
        elif self.up_sample_method == 'transposed':
            self.up_sample = nn.ModuleList(reversed([
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=stage_ch_list[0],
                        out_channels=stage_ch_list[0],
                        kernel_size=(2, 2),
                        stride=(2, 2),
                        bias=bias
                    )
                ) if idx + 1 in down_sample_stages else None for idx in range(num_stages)
            ]))
        elif self.up_sample_method == 'pixel_shuffle':
            self.up_sample = nn.ModuleList(reversed([
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=stage_ch_list[0],
                        out_channels=stage_ch_list[0] * 2 ** 2,
                        kernel_size=(1, 1),
                        bias=bias
                    ),
                    nn.PixelShuffle(
                        upscale_factor=2,
                    ),
                ) if idx + 1 in down_sample_stages else None for idx in range(num_stages)
            ]))
        else:
            ValueError("up_sample_method must be in [bilinear, transposed, pixel_shuffle].")

        self.smoothing_conv_list = nn.ModuleList([
            FeaturePyramidSmoothingConv(
                in_channels=stage_ch_list[0],
                num_sub_bands=2,
                norm_type=norm_type,
                dropout=dropout,
                bias=bias
            ) for _ in range(len(stage_ch_list))
        ])

    def forward(self, f_map_list):
        prev_f_map = None
        f_map_list = list(reversed(f_map_list))

        out_list = []
        for idx, (f_map, down_ch_conv) in enumerate(zip(f_map_list, self.down_ch_conv_list)):

            # Down conv to lowest num channels
            curr_f_map = down_ch_conv(f_map)

            # Initialize prev_f_map
            if prev_f_map is None:
                prev_f_map = curr_f_map
                out = self.smoothing_conv_list[idx](prev_f_map)
                out_list.append(out)
                continue

            if prev_f_map.shape != curr_f_map.shape:
                if self.up_sample_method == 'bilinear':
                    prev_f_map = self.up_sample(prev_f_map)
                else:
                    if self.up_sample[idx] is not None:
                        prev_f_map = self.up_sample[idx](prev_f_map)

            prev_f_map = prev_f_map + curr_f_map
            out = self.smoothing_conv_list[idx](prev_f_map)
            out_list.append(out)

        out = torch.cat(out_list, dim=1)

        return out


class BroadcastPyramidResNet(nn.Module):
    def __init__(self,
                 channels_list: list,
                 in_channels: int,
                 layer_attr: dict = None,
                 feature_norm_type: str = 'res',
                 num_classes: int = 10,
                 up_sample_method: str = 'bilinear',
                 num_two_stage_categorization_class: int = 0,
                 nOut: int = 192,
                 n_mels: int = 80,
                 log_input: bool = True,
                 **kwargs):

        super(BroadcastPyramidResNet, self).__init__()
        self.log_input = log_input
        self.channel_list = channels_list
        if layer_attr is None:
            layer_attr = {
                'bias': False,
                'dropout': 0.1,
            }

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=n_mels),
            )
        self.specaug = FbankAug()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=layer_attr['bias'],
            ),
            Normalize(
                in_channels=channels_list[0],
                norm_type=layer_attr['norm_type'],
            ),
            nn.ReLU(),
            FreqAttnRes2NetUnit(
                num_channels=channels_list[0],
                kernel_size=3,
                padding=1,
                scales=4,
                dropout=0.,
                norm_type=layer_attr['norm_type'],
                bias=layer_attr['bias'],
            ),
        )

        self.stage_1 = nn.Sequential(
            Transition2BlockMod(
                input_dims=channels_list[0],
                output_dims=channels_list[1],
                num_sub_bands=4,
                **layer_attr,
            ),
            Broadcast2BlockMod(
                input_dims=channels_list[1],
                num_sub_bands=4,
                **layer_attr,
            ),
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.feat_norm_1 = None
        if feature_norm_type == 'res':
            self.feat_norm_1 = ResNorm(lam=.1)
        self.freq_ch_attn_1 = FreqAttnRes2NetUnit(
            num_channels=channels_list[1],
            kernel_size=3,
            padding=1,
            scales=4,
            dropout=0.,
            norm_type=layer_attr['norm_type'],
            bias=layer_attr['bias'],
        )

        self.stage_2 = nn.Sequential(
            Transition2BlockMod(
                input_dims=channels_list[1],
                output_dims=channels_list[2],
                num_sub_bands=4,
                **layer_attr,
            ),
            Broadcast2BlockMod(
                input_dims=channels_list[2],
                num_sub_bands=4,
                **layer_attr,
            )
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.feat_norm_2 = None
        if feature_norm_type == 'res':
            self.feat_norm_2 = ResNorm(lam=.1)
        self.freq_ch_attn_2 = FreqAttnRes2NetUnit(
            num_channels=channels_list[2],
            kernel_size=3,
            padding=1,
            scales=4,
            dropout=0.,
            norm_type=layer_attr['norm_type'],
            bias=layer_attr['bias'],
        )

        self.stage_3 = nn.Sequential(
            Transition2BlockMod(
                input_dims=channels_list[2],
                output_dims=channels_list[3],
                num_sub_bands=2,
                **layer_attr,
            ),
            Broadcast2BlockMod(
                input_dims=channels_list[3],
                num_sub_bands=2,
                **layer_attr
            ),
        )
        self.feat_norm_3 = None
        if feature_norm_type == 'res':
            self.feat_norm_3 = ResNorm(lam=.1)
        self.freq_ch_attn_3 = FreqAttnRes2NetUnit(
            num_channels=channels_list[3],
            kernel_size=3,
            padding=1,
            scales=4,
            dropout=0.,
            norm_type=layer_attr['norm_type'],
            bias=layer_attr['bias'],
        )

        self.stage_4 = nn.Sequential(
            Transition2BlockMod(
                input_dims=channels_list[3],
                output_dims=channels_list[4],
                num_sub_bands=2,
                **layer_attr,
            ),
            Broadcast2BlockMod(
                input_dims=channels_list[4],
                num_sub_bands=2,
                **layer_attr
            ),
            Broadcast2BlockMod(
                input_dims=channels_list[4],
                num_sub_bands=2,
                **layer_attr
            ),
        )
        self.feat_norm_4 = None
        if feature_norm_type == 'res':
            self.feat_norm_4 = ResNorm(lam=.1)
        self.freq_ch_attn_4 = FreqAttnRes2NetUnit(
            num_channels=channels_list[4],
            kernel_size=3,
            padding=1,
            scales=4,
            dropout=0.,
            norm_type=layer_attr['norm_type'],
            bias=layer_attr['bias'],
        )

        self.pools_blocks = nn.ModuleList()
        for base in self.channel_list[1:]:  
            bottle_base = int(base/2)
            stats_poolings = AttentiveDoubleStatsPooling(input_dim=base, bottle_dim=bottle_base, kernel=[1,1],  use_global_info=False,
                                                            apply_bn=True, activ="ReLU", normalize_stats=False, use_att_stats_pool=True, use_both_axis=False)    
            self.pools_blocks.append(stats_poolings)

        self.fc = nn.Linear(sum(self.channel_list[1:])*4, nOut)
        self.bn = nn.BatchNorm1d(nOut)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _before_pooling(self, x):
        out = self.stem(x)

        stage_out_list = []

        # Stage 1
        out = self.stage_1(out)
        if self.feat_norm_1 is not None:
            out = self.feat_norm_1(out)
        out = self.freq_ch_attn_1(out)
        stage_out_list.append(out)
        out = self.max_pool_1(out)

        # Stage 2
        out = self.stage_2(out)
        if self.feat_norm_2 is not None:
            out = self.feat_norm_2(out)
        out = self.freq_ch_attn_2(out)
        stage_out_list.append(out)
        out = self.max_pool_2(out)

        # Stage 3
        out = self.stage_3(out)
        if self.feat_norm_3 is not None:
            out = self.feat_norm_3(out)
        out = self.freq_ch_attn_3(out)
        stage_out_list.append(out)

        # Stage 4
        out = self.stage_4(out)
        if self.feat_norm_4 is not None:
            out = self.feat_norm_4(out)
        out = self.freq_ch_attn_4(out)
        stage_out_list.append(out)
        
        return stage_out_list

    def _before_penultimate(self, x_list):
        pool_stats=[]    
        for x, stats_pooling in zip(x_list, self.pools_blocks):

            pool_outs = stats_pooling(x, freq_axis=-2, time_axis=-1)
            pool_stats.append(pool_outs.clone())

        pool_outs = torch.cat(pool_stats, dim=-1)

        x = self.fc(pool_outs)
        x = self.bn(x)
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
                x = x.unsqueeze(dim=1).detach()
        return x

    def wave2emb(self, wave, max_frame=False, aug=False):
        feat = self.wave2feat(wave, max_frame, aug=False)
        late_feat = self._before_pooling(feat)
        emb = self._before_penultimate(late_feat)
        return emb


    def forward(self, x, max_frame=False, aug=False):
        x = self.wave2emb(x, max_frame, aug=False)
        return x


def MainModel(**kwargs):
    # if args.is_large_model:
    #     channel_list = [160, 80, 120, 160, 200]
    # else:
    #     channel_list = [80, 40, 60, 80, 100]

    channel_list = [80, 40, 60, 80, 100]

    model = BroadcastPyramidResNet(
        channels_list=channel_list,
        in_channels=1,
        feature_norm_type='res',
        up_sample_method='pixel_shuffle',
        layer_attr={
            'dropout': 0.1,
            'bias': False,
            'norm_type': 'bn',
        },
        **kwargs
    )

    return model

if __name__=="__main__":
    # batch_size, num_frames, feat_dim = 1, 3000, 80
    batch_size, second = 1, 4
    x = torch.randn(batch_size, int(second*16000))
    # x_tp = x.transpose(1, 2)
    channel_list = [80, 40, 60, 80, 100]
    model = BroadcastPyramidResNet(
        channels_list=channel_list,
        in_channels=1,
        feature_norm_type='res',
        up_sample_method='pixel_shuffle',
        layer_attr={
            'dropout': 0.1,
            'bias': False,
            'norm_type': 'bn',
        }
    )

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
    # self.forward = self.entire_wave2emb ## for use FlopCountAnalysis (Comment out def forward)
    model.eval()
    flops = FlopCountAnalysis(model, x)
    print(flops.total())
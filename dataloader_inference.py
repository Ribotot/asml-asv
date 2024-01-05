#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames=None):
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    # Maximum audio length
    if max_frames == None:
        max_audio = audiosize

        feats = []
        feats.append(audio)

    else:
        max_audio = int(max_frames) * 160 + 240

        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = numpy.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]

        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
        
        feats = []
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat;

def loadWAV_multi2single(filename, channel=0, max_frames=None):
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audio = audio[:,channel]
    audiosize = audio.shape[0]

    # Maximum audio length
    if max_frames == None:
        max_audio = audiosize

        feats = []
        feats.append(audio)

    else:
        max_audio = int(max_frames) * 160 + 240

        if audiosize <= max_audio:
            shortage    = max_audio - audiosize + 1 
            audio       = numpy.pad(audio, (0, shortage), 'wrap')
            audiosize   = audio.shape[0]

        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
        
        feats = []
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat;

class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, **kwargs):
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path,self.test_list[index]))
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)

class test_dataset_loader_dict(Dataset):
    def __init__(self, test_dict, **kwargs):
        self.test_dict  = test_dict
        files           = self.test_dict.keys()
        self.test_list  = list(set(files))
        self.test_list.sort()

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_dict[self.test_list[index]]))
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)

class test_dataset_loader_multichannel(Dataset):
    def __init__(self, test_list, test_path, channel, **kwargs):
        self.test_path  = test_path
        self.test_list  = test_list
        self.channel = channel

    def __getitem__(self, index):
        audio = loadWAV_multi2single(os.path.join(self.test_path,self.test_list[index]), self.channel)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)

class test_dataset_loader_multichannel_dict(Dataset):
    def __init__(self, test_dict, channel, **kwargs):
        self.test_dict  = test_dict
        files           = self.test_dict.keys()
        self.test_list  = list(set(files))
        self.test_list.sort()
        self.channel = channel

    def __getitem__(self, index):
        audio = loadWAV_multi2single(os.path.join(self.test_dict[self.test_list[index]]), self.channel)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)
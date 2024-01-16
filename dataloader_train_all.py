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

    feat = numpy.stack(feats,axis=0).astype(numpy.float64)

    return feat;
    
class AugmentWAV(object):

    def __init__(self, musan_path, sample_noise_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise']

        self.noisesnr   = {'noise':[-3,15]}
        self.numnoise   = {'noise':[1,1]}
        self.clip_prob  = [5, 10]
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        sample_files   = glob.glob(os.path.join(sample_noise_path,'*.wav'));

        for file in sample_files:
            self.noiselist['noise'].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

        self.aug_frame = 60
        self.aug_len = self.aug_frame*160

    def additive_large_noise(self, noisecat, audio, start):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames+self.aug_frame)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
            
        audio = numpy.pad(audio, ((0, 0), (start, self.aug_len-start)), 'constant', constant_values=((0, 0), (0, 0)))

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def additive_small_noise(self, noisecat, audio, start):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = [1, 3]
        noisesnr    = [15, 25]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames+self.aug_frame)
            noise_snr   = random.uniform(noisesnr[0],noisesnr[1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
            
        audio = numpy.pad(audio, ((0, 0), (start, self.aug_len-start)), 'constant', constant_values=((0, 0), (0, 0)))

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


    def audio_clip(self, audio):

        # audio_topk = int(random.uniform(self.clip_prob[0],self.clip_prob[1])/100 * self.max_audio)
        # topk_value  = numpy.min(numpy.partition(numpy.abs(audio), -audio_topk)[:,-audio_topk:])
        topk_value = (1 - random.uniform(self.clip_prob[0],self.clip_prob[1])/100)*numpy.max(numpy.abs(audio))

        return numpy.clip(audio, -topk_value, topk_value)


def make_voxceleb_train_dict(voxceleb_train_list, voxceleb_train_path):
    # Read training files
    with open(voxceleb_train_list) as dataset_file:
        lines = dataset_file.readlines();

    train_dict = {}    
    for line in lines:
        # spk_wavfile = line.strip().split();
        spkutt_wavfile = line.strip().split(); ## if you extract utterances based on kaldi (scp file)

        # spk = spk_wavfile[0]
        spk = spkutt_wavfile[0].split('-')[0] ## if you extract utterances based on kaldi (scp file)

        # filename = os.path.join(voxceleb_train_path,spk_wavfile[-1]);
        filename = os.path.join(voxceleb_train_path,spkutt_wavfile[-1]);
        train_dict[filename] = spk    
    return train_dict

def make_sdsv_train_dict(sdsv_train_path, sdsv_utt2spk_list):
    with open(sdsv_utt2spk_list) as dataset_file:
        lines = dataset_file.readlines();
    utt2spk_dict = {}
    for lidx, line in enumerate(lines):
        if lidx > 0:
            x = line.split()
            utt2spk_dict[x[0]] = x[-1]

    files = glob.glob(os.path.join(sdsv_train_path,'*/*/*.wav'));
    train_dict = {}
    for x in files:
        wavfile = x.strip().split('/')[-1]
        spk_key = wavfile.split('.')[0]
        spk = utt2spk_dict[spk_key]
        train_dict[x] = spk    
    return train_dict

def make_cnceleb_train_dict(cnceleb_train_path):
    files = glob.glob(os.path.join(cnceleb_train_path,'*/*.flac'));
    train_dict = {}
    for x in files:
        spk = x.strip().split('/')[-2]
        train_dict[x] = spk    
    return train_dict

def make_french_mls_train_dict(fre_mls_path):
    files = glob.glob(os.path.join(fre_mls_path,'*/*/*.flac'));
    train_dict = {}
    for x in files:
        spk = x.strip().split('/')[-2]
        train_dict[x] = spk    
    return train_dict

def make_french_tedx_train_dict(fre_tedx_path):
    files = glob.glob(os.path.join(fre_tedx_path,'*/*/*.flac'));
    train_dict = {}
    for x in files:
        spk = x.strip().split('/')[-2]
        train_dict[x] = spk    
    return train_dict

def make_robovox_sample_train_dict(robovox_sample_path):
    files = glob.glob(os.path.join(robovox_sample_path,'*/*.wav'));
    train_dict = {}
    for x in files:
        spk = x.strip().split('/')[-2]
        train_dict[x] = spk    
    return train_dict

class train_dataset_loader(Dataset):
    def __init__(self, voxceleb_train_dict, sdsv_train_dict, cnceleb_train_dict, french_mls_train_dict, french_tedx_train_dict, sample_train_dict, \
        augment_noise, musan_path, robovox_noise_path, rir_path, max_frames, train_path, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, sample_noise_path=robovox_noise_path, rir_path=rir_path, max_frames = max_frames)

        self.max_frames = max_frames;
        self.musan_path = musan_path
        self.sample_noise_path = robovox_noise_path
        self.rir_path   = rir_path
        self.augment_noise    = augment_noise
        
        self.data_list  = []
        self.data_label = []
        spk_bias = 0

        voxceleb_spks = list(set(voxceleb_train_dict.values()))
        voxceleb_spks.sort()
        voxceleb_spk_dict = {}
        for lidx, spk_lab in enumerate(voxceleb_spks):
            voxceleb_spk_dict[spk_lab] = lidx
        for filename, spk in voxceleb_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(voxceleb_spk_dict[spk])
        spk_bias += len(voxceleb_spks)
        print('VoxCeleb2 Speaker : ',len(voxceleb_spks))

        sdsv_spks = list(set(sdsv_train_dict.values()))
        sdsv_spks.sort()
        sdsv_spk_dict = {}
        for lidx, spk_lab in enumerate(sdsv_spks):
            sdsv_spk_dict[spk_lab] = lidx + spk_bias
        for filename, spk in sdsv_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(sdsv_spk_dict[spk])     
        spk_bias += len(sdsv_spks)
        print('SDSV2020 task2 Speaker : ',len(sdsv_spks))

        cnceleb_spks = list(set(cnceleb_train_dict.values()))
        cnceleb_spks.sort()
        cnceleb_spk_dict = {}
        for lidx, spk_lab in enumerate(cnceleb_spks):
            cnceleb_spk_dict[spk_lab] = lidx + spk_bias
        for filename, spk in cnceleb_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(cnceleb_spk_dict[spk])
        spk_bias += len(cnceleb_spks)
        print('CN-Celeb1 Speaker : ',len(cnceleb_spks))

        french_mls_spks = list(set(french_mls_train_dict.values()))
        french_mls_spks.sort()
        french_mls_spk_dict = {}
        for lidx, spk_lab in enumerate(french_mls_spks):
            french_mls_spk_dict[spk_lab] = lidx + spk_bias
        for filename, spk in french_mls_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(french_mls_spk_dict[spk])
        spk_bias += len(french_mls_spks)
        print('Multilingual LibriSpeech (French) Speaker : ',len(french_mls_spks))

        french_tedx_spks = list(set(french_tedx_train_dict.values()))
        french_tedx_spks.sort()
        french_tedx_spk_dict = {}
        for lidx, spk_lab in enumerate(french_tedx_spks):
            french_tedx_spk_dict[spk_lab] = lidx + spk_bias
        for filename, spk in french_tedx_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(french_tedx_spk_dict[spk])
        spk_bias += len(french_tedx_spks)
        print('Multilingual TEDx (French) Speaker : ',len(french_tedx_spks))

        sample_spks = list(set(sample_train_dict.values()))
        sample_spks.sort()
        sample_spk_dict = {}
        for lidx, spk_lab in enumerate(sample_spks):
            sample_spk_dict[spk_lab] = lidx + spk_bias
        for filename, spk in sample_train_dict.items():
            self.data_list.append(filename)
            self.data_label.append(sample_spk_dict[spk])
        spk_bias += len(sample_spks)
        print('ROBOVOX Sample Speaker : ',len(sample_spks))
        print('Total Speaker :',spk_bias)

        self.num_label = spk_bias

    def __getitem__(self, indices):

        feat = []
        #z#

        for index in indices:
            audio = loadWAV(self.data_list[index], self.max_frames)
            
            if self.augment_noise:
                augtype = random.random()
                start = random.randint(1, 9599) #self.aug_len
                if augtype > 0.8:
                    audio   = self.augment_wav.reverberate(audio)
                    audio   = self.augment_wav.additive_large_noise('noise',audio,start)
                elif augtype > 0.6:
                    audio   = self.augment_wav.reverberate(audio)
                    audio   = self.augment_wav.additive_small_noise('noise',audio,start)
                elif augtype > 0.5:
                    audio   = self.augment_wav.audio_clip(audio)
                    audio   = self.augment_wav.additive_large_noise('noise',audio,start)
                elif augtype > 0.3:
                    audio   = self.augment_wav.audio_clip(audio)
                    audio   = self.augment_wav.additive_small_noise('noise',audio,start)
                elif augtype > 0.1:
                    audio   = self.augment_wav.additive_large_noise('noise',audio,start)
                else:
                    audio   = self.augment_wav.additive_small_noise('noise',audio,start)                            

            feat.append(audio);

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_list)


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label;
        self.nPerSpeaker        = nPerSpeaker;
        self.max_seg_per_spk    = max_seg_per_spk;
        self.batch_size         = batch_size;
        self.epoch              = 0;
        self.seed               = seed;
        self.distributed        = distributed;
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);


        ## Group file indices for each class
        dictkeys = list(data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)

            
            rp      = lol(numpy.arange(numSeg),self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


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


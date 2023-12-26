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
    
class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

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
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]


class train_dataset_loader(Dataset):
    def __init__(self, train_list, augment_noise, musan_path, rir_path, max_frames, train_path, **kwargs):

        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.train_list = train_list
        self.max_frames = max_frames;
        self.musan_path = musan_path
        self.rir_path   = rir_path
        self.augment_noise    = augment_noise
        
        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines();

        # Make a dictionary of ID names and ID indices
        ## Please note that, if you extract utterances based on kaldi (scp file), you will get spk-video-utt. This code is used for kaldi-based scp file. ##
        dictkeys = list(set([x.split()[0].split('-')[0] for x in lines]))
        ## Please use the folloing code. ##
        # dictkeys = list(set([x.split()[0] for x in lines]))

        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split();

            ## Please note that, if you extract utterances based on kaldi (scp file), you will get spk-video-utt. This code is used for kaldi-based scp file. ##
            speaker_label = dictkeys[data[0].split('-')[0]];
            ## Please use the folloing code. ##
            # speaker_label = dictkeys[data[0]];

            filename = os.path.join(train_path,data[1]);
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):

        feat = []
        #z#

        for index in indices:
            audio = loadWAV(self.data_list[index], self.max_frames)
            
            if self.augment_noise:
                RIR_augtype = random.random()    
                if RIR_augtype > 0.5:
                    audio   = self.augment_wav.reverberate(audio)

                Noise_augtype = random.random()
                if Noise_augtype > 0.85:
                    audio   = self.augment_wav.additive_noise('music',audio)
                elif Noise_augtype > 0.7:
                    audio   = self.augment_wav.additive_noise('speech',audio)
                elif Noise_augtype > 0.5:
                    audio   = self.augment_wav.additive_noise('noise',audio)
                

            feat.append(audio);

        feat = numpy.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_list)



class test_dataset_loader(Dataset):
    def __init__(self, test_list, test_path, eval_frames=None, **kwargs):
        self.eval_frames = eval_frames;
        self.test_path  = test_path
        self.test_list  = test_list

    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path,self.test_list[index]), self.eval_frames)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


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



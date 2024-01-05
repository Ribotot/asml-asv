#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, numpy, sys, random
import time, itertools, importlib

from dataloader_inference import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from utils import save_on_master, load_dict_txt, load_dict_npy, numpy_normalize

from kaldi_io.meta_utils import gather_scps, make_utt2spk, make_scp, prepend_lines
from kaldi_io.arkio import ArchiveReader, read_kaldi_binary, write_kaldi_binary

class ModelInferencer(object):
    def __init__(self, speaker_model, optimizer, gpu, enroll_frames, test_frames, crop_option, **kwargs):

        self.__model__ = speaker_model

        self.gpu = gpu

        self.enroll_frames = enroll_frames
        self.test_frames = test_frames
        self.crop_option = crop_option

        if enroll_frames == test_frames:
            self.equal_len = True
        else:
            self.equal_len = False


    def crop(self, wav, eval_frames=None, crop_option='F', sample_rate=160):
        if eval_frames == None:
            return wav
        else:
            nframes = wav.size()[-1]
            eval_frames = eval_frames*sample_rate
            if nframes < eval_frames:
                eval_frames = nframes

            if crop_option == 'F': 
                wav = wav[..., :eval_frames]
            elif crop_option == 'C':
                nframes = wav.size()[-2]
                start = max(nframes//2 - 1 - eval_frames//2, 0)
                wav = wav[..., start:start+eval_frames]
            elif crop_option == 'R': 
                wav = wav[..., -eval_frames:]
            else: 
                raise ValueError('--crop option must be "F", "C", or "R"!')
            return wav

    def lab_transf(self, line):
        if line == 'imp':
            return 0
        elif line == 'tgt':
            return 1
        else:
            raise ValueError('Undefined labeled type')

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_trial, enroll_list, test_list, test_path,  \
        nDataLoaderThread, distributed, print_interval=100, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        feats_enr = {}
        feats_te = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Read enroll list file as lines_enr and make dict
        dict_enr = {}
        with open(enroll_list) as f:
            lines_enr = f.readlines()
            for x in lines_enr:
                dict_enr[x.strip().split()[0]] = x.strip().split()[1]
        files = list(dict_enr.values())
        files_enr = list(set(files))
        ## Read test_list file
        files = list([x.strip().split()[1] for x in lines])
        files_te = list(set(files)) 
        if test_trial == 'eval':            
            for _ in range(len(files_te)):
                new_line = 'sid_eval/'+files_te.pop(0)
                files_te.append(new_line)

        ## Get a list of unique file names
        setfiles = list(set(files_enr+files_te))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every utterance
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat_enr = self.__model__.module.inference(self.crop(inp1,self.enroll_frames,self.crop_option)).detach().cpu()
            feats_enr[data[1][0]] = ref_feat_enr
            if not self.equal_len:
                with torch.no_grad():
                    ref_feat_te = self.__model__.module.inference(self.crop(inp1,self.test_frames,self.crop_option)).detach().cpu()
                feats_te[data[1][0]] = ref_feat_te
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat_enr.size()[1])
                )
        if self.equal_len:
            feats_te = feats_enr

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all_enr = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all_enr, feats_enr)
            feats_all_te = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all_te, feats_te)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats_enr = feats_all_enr[0]
                for feats_batch in feats_all_enr[1:]:
                    feats_enr.update(feats_batch)
                feats_te = feats_all_te[0]
                for feats_batch in feats_all_te[1:]:
                    feats_te.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()
                if test_trial == 'eval':
                    data[1] = 'sid_eval/'+data[1]

                ## Append random label if missing
                if len(data) == 2:
                    data = data + [random.randint(0, 1)]

                ref_feat = feats_enr[dict_enr[data[0]]].cuda()
                com_feat = feats_te[data[1]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
                all_scores.append(score)
                all_labels.append(int(self.lab_transf(data[2])))
                all_trials.append(data[0] + " " + data[1])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list with score normalize (adaptive s-norm)
    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList_easy_aSn(self, test_trial, enroll_list, test_list, test_path, \
        nDataLoaderThread, distributed, coh_size=100, print_interval=100, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        feats_enr = {}
        feats_te = {}
        means_enr = {}
        means_te = {}
        stds_enr  = {}
        stds_te  = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Read enroll list file as lines_enr and make dict
        dict_enr = {}
        with open(enroll_list) as f:
            lines_enr = f.readlines()
            for x in lines_enr:
                dict_enr[x.strip().split()[0]] = x.strip().split()[1]
        files = list(dict_enr.values())
        files_enr = list(set(files))
        ## Read test_list file
        files = list([x.strip().split()[1] for x in lines])
        files_te = list(set(files)) 
        if test_trial == 'eval':            
            for _ in range(len(files_te)):
                new_line = 'sid_eval/'+files_te.pop(0)
                files_te.append(new_line)

        ## Get a list of unique file names
        setfiles = list(set(files_enr+files_te))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every utterance
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat_enr = self.__model__.module.inference(self.crop(inp1,self.enroll_frames,self.crop_option))
                cosine = F.linear(F.normalize(ref_feat_enr), F.normalize(self.__model__.module.__L__.weight))
                cohort, _ = torch.topk(cosine, coh_size, dim=-1, largest=True, sorted=True)
                var, mean = torch.var_mean(cohort, dim=-1, keepdims=True)
                std = torch.sqrt(var)
            feats_enr[data[1][0]] = ref_feat_enr.detach().cpu()
            means_enr[data[1][0]] = mean.detach().cpu()
            stds_enr[data[1][0]]  = std.detach().cpu()
            if not self.equal_len:
                with torch.no_grad():
                    ref_feat_te = self.__model__.module.inference(self.crop(inp1,self.test_frames,self.crop_option))
                    cosine = F.linear(F.normalize(ref_feat_te), F.normalize(self.__model__.module.__L__.weight))
                    cohort, _ = torch.topk(cosine, coh_size, dim=-1, largest=True, sorted=True)
                    var, mean = torch.var_mean(cohort, dim=-1, keepdims=True)
                    std = torch.sqrt(var)                    
                feats_te[data[1][0]] = ref_feat_te.detach().cpu()
                means_te[data[1][0]] = mean.detach().cpu()
                stds_te[data[1][0]]  = std.detach().cpu()
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat_enr.size()[1])
                )
        
        if self.equal_len:
            feats_te = feats_enr
            means_te = means_enr
            stds_te = stds_enr

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all_enr = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all_enr, feats_enr)
            feats_all_te = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all_te, feats_te)
        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats_enr = feats_all_enr[0]
                for feats_batch in feats_all_enr[1:]:
                    feats_enr.update(feats_batch)
                feats_te = feats_all_te[0]
                for feats_batch in feats_all_te[1:]:
                    feats_te.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()
                if test_trial == 'eval':
                    data[1] = 'sid_eval/'+data[1]

                ## Append random label if missing
                if len(data) == 2:
                    data = data + [random.randint(0, 1)]

                ref_feat = feats_enr[dict_enr[data[0]]].cuda()
                com_feat = feats_te[data[1]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                score = torch.mean(torch.matmul(ref_feat, com_feat.T)) # Get the score
                score = 0.5*(score-means_enr[dict_enr[data[0]]].cuda()[0])/stds_enr[dict_enr[data[0]]].cuda()[0] \
                         + 0.5*(score-means_te[data[1]].cuda()[0])/stds_te[data[1]].cuda()[0]
                score = numpy.mean(score.detach().cpu().numpy())

                all_scores.append(score)
                all_labels.append(int(self.lab_transf(data[2])))
                all_trials.append(data[0] + " " + data[1])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluation with speaker embedding saving (only testset)
    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList_saveSE(self, test_trial, enroll_list, test_list, test_path,  \
        nDataLoaderThread, distributed, result_save_path, epoch, print_interval=100, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()
 
        if self.equal_len:
            feats_enr = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Read enroll list file as lines_enr and make dict
        dict_enr = {}
        with open(enroll_list) as f:
            lines_enr = f.readlines()
            for x in lines_enr:
                dict_enr[x.strip().split()[0]] = x.strip().split()[1]
        files = list(dict_enr.values())
        files_enr = list(set(files))
        ## Read test_list file
        files = list([x.strip().split()[1] for x in lines])
        files_te = list(set(files)) 
        if test_trial == 'eval':            
            for _ in range(len(files_te)):
                new_line = 'sid_eval/'+files_te.pop(0)
                files_te.append(new_line)
            trial_type = '/voices2019_eval'
        else:
            trial_type = '/voices2019_dev'

        ## Get a list of unique file names
        setfiles = list(set(files_enr+files_te))
        setfiles.sort()

        SE_save_path    = result_save_path+"/Epoch"+str(epoch)+trial_type
        os.makedirs(SE_save_path, exist_ok=True)

        if self.enroll_frames == None:
            arkfile_enr = SE_save_path+'/embedding_enr.ark'
        else:
            arkfile_enr = SE_save_path+'/embedding_enr_{:d}{:s}.ark'.format(self.enroll_frames, self.crop_option)
        scpfile_enr = arkfile_enr.replace('.ark', '.scp')
        if not os.path.isfile(arkfile_enr):
            ## Define test data loader
            test_dataset = test_dataset_loader(setfiles, test_path, **kwargs)

            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            else:
                sampler = None

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

            with open(arkfile_enr, 'wb') as f:
                ## Extract features for every utterance
                for idx, data in enumerate(test_loader):
                    inp1 = data[0][0].cuda()
                    with torch.no_grad():
                        ref_feat_enr = self.__model__.module.inference(self.crop(inp1,self.enroll_frames,self.crop_option)).detach().cpu()
                    write_kaldi_binary(f, array=ref_feat_enr.numpy(), key=data[1][0])
                    if self.equal_len:
                        feats_enr[data[1][0]] = ref_feat_enr.numpy()
                    telapsed = time.time() - tstart

                    if idx % print_interval == 0 and rank == 0:
                        sys.stdout.write(
                            "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat_enr.size()[1])
                        )
            print("\nEnrolled speaker embedding extraction completed.")
        else:
            print("\nEnrolled speaker embedding extraction has already been completed.")

        if not os.path.isfile(scpfile_enr):
            make_scp(arkfile_enr, scp_out=scpfile_enr)

        utt2arkp_enr = load_dict_txt(scpfile_enr)

        tstart = time.time()

        if self.test_frames == None:
            arkfile_te = SE_save_path+'/embedding_te.ark'
        else:
            arkfile_te = SE_save_path+'/embedding_te_{:d}{:s}.ark'.format(self.test_frames, self.crop_option)
        scpfile_te = arkfile_te.replace('.ark', '.scp')
        if self.equal_len:
            if not os.path.isfile(arkfile_te):
                with open(arkfile_te, 'wb') as f:
                    for key, var in feats_enr.items():
                        write_kaldi_binary(f, array=var, key=key)
                print("Tested speaker embedding extraction completed.")
            else:
                print("Tested speaker embedding extraction has already been completed.")
        else:
            if not os.path.isfile(arkfile_te):
                ## Define test data loader
                test_dataset = test_dataset_loader(setfiles, test_path, **kwargs)

                if distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
                else:
                    sampler = None

                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

                with open(arkfile_te, 'wb') as f:
                    ## Extract features for every utterance
                    for idx, data in enumerate(test_loader):
                        inp1 = data[0][0].cuda()
                        with torch.no_grad():
                            ref_feat_te = self.__model__.module.inference(self.crop(inp1,self.test_frames,self.crop_option)).detach().cpu()
                        write_kaldi_binary(f, array=ref_feat_te.numpy(), key=data[1][0])
                        telapsed = time.time() - tstart

                        if idx % print_interval == 0 and rank == 0:
                            sys.stdout.write(
                                "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat_te.size()[1])
                            )
                print("\nTested speaker embedding extraction completed.")
            else:
                print("\nTested speaker embedding extraction has already been completed.")

        if not os.path.isfile(scpfile_te):
            make_scp(arkfile_te, scp_out=scpfile_te)

        utt2arkp_te = load_dict_txt(scpfile_te)

        all_scores = []
        all_labels = []
        all_trials = []

        if rank == 0:

            tstart = time.time()
            print("")

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()
                if test_trial == 'eval':
                    data[1] = 'sid_eval/'+data[1]

                ## Append random label if missing
                if len(data) == 2:
                    data = data + [random.randint(0, 1)]

                ref_feat = read_kaldi_binary(utt2arkp_enr[dict_enr[data[0]]])
                com_feat = read_kaldi_binary(utt2arkp_te[data[1]])
                
                if self.__model__.module.__L__.test_normalize:
                    ref_feat = numpy_normalize(ref_feat, p=2, dim=1)
                    com_feat = numpy_normalize(com_feat, p=2, dim=1)

                score = numpy.mean(numpy.matmul(ref_feat, com_feat.T)) # Get the score
                all_scores.append(score)
                all_labels.append(int(self.lab_transf(data[2])))
                all_trials.append(data[0] + " " + data[1])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)



    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        checkpoint = torch.load(path, map_location="cpu")
        print("Ckpt file %s loaded!"%(path))

        if 'network' not in checkpoint.keys():
            loaded_state = checkpoint
        else:
            loaded_state = checkpoint['network']

        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

        print("Model loaded!")
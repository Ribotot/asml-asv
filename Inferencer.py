#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, numpy, sys, random
import time, itertools, importlib

from dataloader_voxceleb import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from utils import save_on_master, load_dict_txt, load_dict_npy, numpy_normalize

from kaldi_io.meta_utils import gather_scps, make_utt2spk, make_scp, prepend_lines
from kaldi_io.arkio import ArchiveReader, read_kaldi_binary, write_kaldi_binary

class ModelInferencer(object):
    def __init__(self, speaker_model, optimizer, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        self.gpu = gpu

        self.mixedprec = mixedprec

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
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
                ref_feat = self.__model__.module.inference(inp1).detach().cpu()
            feats[data[1][0]] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):

                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                score = numpy.mean(torch.matmul(ref_feat, com_feat.T).detach().cpu().numpy()) # Get the score
                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluation with speaker embedding saving (only testset)
    ## ===== ===== ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList_saveSE(self, test_list, test_path, nDataLoaderThread, distributed, result_save_path, epoch, print_interval=100, **kwargs):

        SE_save_path    = result_save_path+"/Epoch"+str(epoch)
        os.makedirs(SE_save_path, exist_ok=True)

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        arkfile_te = SE_save_path+'/embedding.ark'
        scpfile_te = arkfile_te.replace('.ark', '.scp')
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
                        ref_feat = self.__model__.module.inference(inp1).detach().cpu()
                    write_kaldi_binary(f, array=ref_feat.numpy(), key=data[1][0])
                    telapsed = time.time() - tstart

                    if idx % print_interval == 0 and rank == 0:
                        sys.stdout.write(
                            "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                        )
            print("Speaker embedding extraction completed.")
        else:
            print("Speaker embedding extraction has already been completed.")

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

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = read_kaldi_binary(utt2arkp_te[data[1]])
                com_feat = read_kaldi_binary(utt2arkp_te[data[2]])
                
                if self.__model__.module.__L__.test_normalize:
                    ref_feat = numpy_normalize(ref_feat, p=2, dim=1)
                    com_feat = numpy_normalize(com_feat, p=2, dim=1)

                score = numpy.mean(numpy.matmul(ref_feat, com_feat.T)) # Get the score
                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

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
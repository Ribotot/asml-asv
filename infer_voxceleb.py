#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse
import yaml
import numpy
import torch
import glob
import zipfile
import warnings
import datetime
from tuneThreshold import *
from SpeakerNet import *
from dataloader_voxceleb import *
warnings.simplefilter("ignore")

## ===== ===== ===== ===== ===== ===== ===== =====
## Parse arguments
## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "SpeakerNet")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions')
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.01,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## For test only
## Test data
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(gpu, ngpus_per_node, args):


    ## Load models
    s = SpeakerNet(**vars(args))

    s = WrappedModel(s).cuda(gpu)

    it = 1

    ## Write args to scorefile
    scorefile   = open(args.result_save_path+"/scores.txt", "a+")

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## Evaluation code - must run on single GPU

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())

    print('Total parameters: ',pytorch_total_params)
    print('Test list',args.test_list)
    
    sc, lab, _ = trainer.evaluateFromList(**vars(args))

    result = tuneThresholdfromScore(sc, lab, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

    print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "VEER {:2.4f}".format(result[1]), "MinDCF {:2.5f}".format(mindcf))


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    gpu = args.gpu

    main_worker(gpu, None, args)


if __name__ == '__main__':
    main()
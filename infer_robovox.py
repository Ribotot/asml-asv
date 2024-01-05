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
from dataloader_inference import *
from Inferencer_robovox import *
from SpeakerNet import WrappedModel, SpeakerNet
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
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.01,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')
parser.add_argument('--eval_type',  type=str,   default=None,   choices=["save_te", "easy_sn", None], help='Initial load_type')
parser.add_argument('--coh_size',       type=int,   default=400,    help='Cohorts size for score normalization')


## Load and save
parser.add_argument('--epoch',          type=int,   default=-1,     help='Load model weights of epoch')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Test option
parser.add_argument('--enroll_frames',  type=int,   default=None,    help='Input length to the network for training')
parser.add_argument('--test_frames',    type=int,   default=None,    help='Input length to the network for training')
parser.add_argument('--crop_option',    type=str,   default=None,   choices=['F', 'C', 'R'], help='Initial crop_option. F, C, and R denote "front", "center", and "random", respectively')

## Test data (For test only)
parser.add_argument('--test_trial',     type=str,   default="multi",   choices=['multi', 'single'], help='Initial test_trial')
parser.add_argument('--channel',        type=int,   default=4,   choices=[0,1,2,3,4,5,6,7], help='Select channel for Robovox multichannel task')
# parser.add_argument('--dev_list',       type=str,   default="data/test_list_multi.txt",   help='Robovox multichannel task trial list')
# parser.add_argument('--eval_list',      type=str,   default="data/test_list_single.txt",   help='Robovox singlechannel task trial list')
# parser.add_argument('--dev_path',       type=str,   default="data/robovox_sp_cup_2024/trian",   help='Absolute path to the Robovox multichannel set')
# parser.add_argument('--eval_path',      type=str,   default="data/robovox_sp_cup_2024/test",    help='Absolute path to the Robovox singlechannel set')
    ########## real example (Choi Jeong-Hwan) ########### 
parser.add_argument('--dev_list',       type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/ROBOVOX_SP_CUP_2024/data/multi-channel/multi-channel-trials.trl",     help='VOiCES2019 dev trial list')
parser.add_argument('--eval_list',      type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/ROBOVOX_SP_CUP_2024/data/single-channel/signle-channel-trials.trl",     help='VOiCES2019 eval trial list')
parser.add_argument('--dev_path',       type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/ROBOVOX_SP_CUP_2024/data/multi-channel/",   help='Absolute path to the VOiCES2019 dev set')
parser.add_argument('--eval_path',      type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/ROBOVOX_SP_CUP_2024/data/single-channel/", help='Absolute path to the VOiCES2019 eval set')


## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,     help='Stride size of the first analytic filterbank layer of RawNet3')
parser.add_argument('--gpu_id',         type=str,   default="0",    help='GPU')


## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')


## Participants options
parser.add_argument('--submitted',      dest='submitted',   action='store_true', help='Gernerated the submission file')

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

    s = WrappedModel(s).cuda()
    
    it = 1

    args.gpu = args.gpu_id

    if args.test_trial == "multi":
        args.test_list   = args.dev_list
        args.test_path   = args.dev_path
        trial_type = '/robovox2024_multi'
    elif args.test_trial == "single":
        args.test_list   = args.eval_list
        args.test_path   = args.eval_path
        trial_type = '/robovox2024_single'
    else:
        raise ValueError('Undefined test trial type')    

    trainer     = ModelInferencer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.epoch != -1):
        for model in modelfiles:
            epoch = model.strip().split('.')[0].split('model')[2]
            if int(epoch) == args.epoch:
                init_model = model
        trainer.loadParameters(init_model)
        print("Model epoch {} loaded!".format(args.epoch))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## Evaluation code - must run on single GPU

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())/ 1000 / 1000
    pytorch_loss_params = sum(p.numel() for p in s.module.__L__.parameters())/ 1000 / 1000

    print('Model parameters: {:.4f} M'.format(pytorch_total_params))
    print('Loss part parameters: {:.4f} M'.format(pytorch_loss_params))
    print('Total parameters: {:.4f} M'.format(pytorch_total_params + pytorch_loss_params))
    print('Enrollment frames: ',args.enroll_frames, 'Test frames: ',args.test_frames, 'Crop: ',args.crop_option)
    print('Test trial type: ',args.test_trial)
    print('Test list: ',args.test_list)

    
    if args.eval_type == None:
        sc, lab, tri = trainer.evaluateFromList(**vars(args))
    elif args.eval_type == "save_te":
        sc, lab, tri = trainer.evaluateFromList_saveSE(**vars(args))
    elif args.eval_type == "easy_sn":
        sc, lab, tri = trainer.evaluateFromList_easy_aSn(**vars(args))
    else:
        raise ValueError('Undefined evaluate type')    

    result = tuneThresholdfromScore(sc, lab, [1, 0.1])

    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
    day_mindcf, day_threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.01, 10, 100)
    night_mindcf, night_threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.8, 1, 20)

    final_save_path = args.result_save_path+"/Epoch"+str(args.epoch)+trial_type
    os.makedirs(final_save_path, exist_ok=True)
    scorefile   = open(final_save_path+"/results.txt", "a+")

    print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}".format(args.epoch), \
        "VEER {:2.4f}".format(result[1]), "MinDCF[{:2.3f}]".format(args.dcf_p_target) ,"{:2.5f}".format(mindcf), \
        "MinDCF_Day", "{:2.5f}".format(day_mindcf), "MinDCF_Night", "{:2.5f}".format(night_mindcf))
    scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF[{:2.3f}] {:2.5f}, MinDCF_Day {:2.5f}, MinDCF_Night {:2.5f}\n".format(args.epoch, result[1], args.dcf_p_target, mindcf, day_mindcf, night_mindcf))
    scorefile.close()

    tri_resultfile = open(final_save_path+"/trial_result.txt", "w+")
    only_scorefile = open(final_save_path+"/only_score.txt", "w+")
    if args.submitted:
        print('Gernerated the submission file\n')
        submision_file = open(final_save_path+"/submission.txt", "w+")
        if args.eval_type == "easy_sn":
            sc = min_max_norm(sc)

    for score, target, line in zip(sc, lab, tri):
        tri_resultfile.write(line+' {:3.3f}'.format(score) +' {:3.3f}\n'.format(target))
        only_scorefile.write('{:3.3f}\n'.format(score))
        if args.submitted:
            data = line.strip().split()
            submision_file.write(data[0]+'\t'+data[1]+'\t{:3.7f}\n'.format(1.0 - score))
    tri_resultfile.close()
    only_scorefile.close()

def min_max_norm(input_):
    min_value = min(input_)
    max_value = max(input_)
    scale = max_value - min_value
    output = (input_- min_value) / scale
    # print(output[0:10])
    # output = []
    # for line in input_:
    #     output.append((line - min_value) / scale)
    # print(output[0:10])
    # exit()

    return output


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====


def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    main_worker(0, None, args)


if __name__ == '__main__':
    main()
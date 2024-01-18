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
from Trainer import *
from dataloader_train2 import *
from utils import *
from SpeakerNet import WrappedModel, SpeakerNet
import torch.distributed as dist
import torch.multiprocessing as mp
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
parser.add_argument('--nDataLoaderThread', type=int, default=6,     help='Number of loader threads')
parser.add_argument('--augment_noise',  type=bool,  default=False,  help='Augment noise and RIR to input')
parser.add_argument('--augment_specaug',type=bool,  default=False,  help='Augment specaugmentation to input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,     help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=80,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="",    help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="multisteplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.5,    help='Learning rate decay')
parser.add_argument('--clip_grad',      type=float, default=3.0,    help="""Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. 0 for disabling.""");
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer')
parser.add_argument("--decay_interval", type=int,   default=10,     help='Learning rate decay interval, only for [steplr] scheduler')
parser.add_argument("--decay_epochs",   type=str,   default='[40, 45, 50, 55, 60, 65, 70, 75]', help='Learning rate decay epochs, only for [multisteplr] scheduler')
parser.add_argument("--warmup_epoch",   type=int,   default=10,     help='Learning rate decay epochs, only for [onecyclelr] scheduler')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for [triplet] loss functions')
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for [triplet] loss functions')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.01,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs')

## Training and test data
# parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list')
# parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list')
# parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set')
# parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set')
# parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set')
# parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')
    ########## real example (Choi Jeong-Hwan) ########### 
parser.add_argument('--train_list',        type=str,   default="/home/jh2/Workspace/cjh/fire/sess_torch/meta_vc2_mfbe80/voxceleb2/all/wav_vc2_train.scp",           help='Path for Vox_2 dev list')
# parser.add_argument('--train_list',        type=str,   default="/home/jh2/Workspace/cjh/fire/sess_torch/meta_vc2_mfbe80/voxceleb2/all/wav_vc_all_train.scp",           help='Path for Vox_2 dev list')
parser.add_argument('--test_list',         type=str,   default="/home/jh2/Workspace/cjh/fire/sess_torch/meta_vc1/veri_test_clean.txt",     help='Evaluation list');
parser.add_argument('--train_path',        type=str,   default="",          help='Absolute path to the train set');
parser.add_argument('--test_path',         type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/VoxCeleb1/test/wav/", help='Absolute path to the test set');
parser.add_argument('--musan_path',        type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',          type=str,   default="/media/jh2/f22b587f-8065-4c02-9b74-f6b9f5a89581/DB/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--sinc_stride',    type=int,   default=10,    help='Stride size of the first analytic filterbank layer of RawNet3')
parser.add_argument('--gpu_id',         type=str,   default="0",    help='Select GPU')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

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

    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port

        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)

        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)

        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)

        print('Loaded the model on GPU {:d}'.format(args.gpu))

    else:
        s = WrappedModel(s).cuda()

    it = 1
    eers = [100]

    args.gpu = args.gpu_id

    if args.gpu == args.gpu_id:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")
        dict2scp(args.result_save_path+"/args.scp", vars(args))

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))

    train_sampler = train_dataset_sampler(train_dataset, **vars(args))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    if args.scheduler == 'onecyclelr':
        epoch_per_sample = []
        for _ in range(1):
            iter(train_sampler)
            epoch_per_sample.append(len(train_sampler))
        args.epoch_per_sample = max(epoch_per_sample)
    
    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()

    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model, load_optimizer=False, load_scheduler=False)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## Save training code and params
    if args.gpu == args.gpu_id:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()

        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):

        train_sampler.set_epoch(it)

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        loss, traineer = trainer.train_network(train_loader, verbose=(args.gpu == args.gpu_id))

        if args.gpu == args.gpu_id:
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f}".format(it, traineer, loss, max(clr)))
            scorefile.write("Epoch {:d}, TEER/TAcc {:2.2f}, TLOSS {:f}, LR {:f} \n".format(it, traineer, loss, max(clr)))

        if it % args.test_interval == 0:

            sc, lab, _ = trainer.evaluateFromList(**vars(args))

            if args.gpu == args.gpu_id:
                
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])

                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)

                eers.append(result[1])

                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}".format(it, result[1], mindcf))
                scorefile.write("Epoch {:d}, VEER {:2.4f}, MinDCF {:2.5f}\n".format(it, result[1], mindcf))

                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)

                with open(args.model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
                    eerfile.write('{:2.4f}'.format(result[1]))

                scorefile.flush()

    if args.gpu == args.gpu_id:
        scorefile.close()


## ===== ===== ===== ===== ===== ===== ===== =====
## Main function
## ===== ===== ===== ===== ===== ===== ===== =====

def main():
    args.model_save_path     = args.save_path+"/model"
    args.result_save_path    = args.save_path+"/result"
    args.feat_save_path      = ""

    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)

    # n_gpus = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == '__main__':
    main()
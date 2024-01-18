#!/bin/bash

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM2.yaml \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml

python train_select_gpu.py \
--config ./configs/ECAPATDNN2048_AAM.yaml \
--main_gpu_id 0 \
--gpus 0 \
--initial_model /home/jh2/Downloads/asml-asv/exps/ECAPATDNN2048_AAM_old/model/model000000024.model \
--decay_epochs '[30, 35, 40, 45, 50, 55, 60, 55]' \
--max_epoch 70 \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM5.yaml \

# python DEV_train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM2.yaml

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN_Light_AAM.yaml \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --distributed

# python train_voxceleb.py \
# --config ./configs/ECAPA2_ASML_AAM.yaml

# python infer_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --epoch 75 \
# --eval_type easy_sn\

# python infer_robovox.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 77 --test_trial multi --gpu_id 0
# python infer_robovox.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 77 --eval_type easy_sn --test_trial multi --gpu_id 0

# python infer_voxceleb.py --config ./configs/ECAPA2_ASML_AAM.yaml --epoch 79 --eval_type save_te --test_trial H --gpu_id 1 --enroll_frames 1000 --test_frames 200 --crop_option C
# python infer_voices.py --config ./configs/ECAPA2_ASML_AAM.yaml --epoch 79 --eval_type save_te --test_trial dev --gpu_id 1 --enroll_frames 1000 --test_frames 200 --crop_option C
# python infer_voices.py --config ./configs/ECAPA2_ASML_AAM.yaml --epoch 79 --eval_type save_te --test_trial eval --gpu_id 1 --enroll_frames 1000 --test_frames 200 --crop_option C
#!/bin/bash

python train_voxceleb.py \
--config ./configs/ECAPATDNN512_AAM.yaml \

# python DEV_train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM2.yaml

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN_Light_AAM.yaml \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --distributed

# python train_voxceleb.py \
# --config ./configs/ResNetSE34L_AAM.yaml

# python infer_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --epoch 75 \
# --eval_type easy_sn\

# python infer_voxceleb.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 14 --eval_type save_te --gpu_id 1
# python infer_voxceleb.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 14 --eval_type save_te --test_trial O --gpu_id 1 --enroll_frames 200 --test_frames 200 --crop_option C
# python infer_voxceleb.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 71 --eval_type save_te --test_trial O --gpu_id 1
# python infer_voices.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 75 --test_trial dev --gpu_id 1 --enroll_frames 1000 --test_frames 200 --crop_option C
# python infer_robovox.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 75 --test_trial multi --gpu_id 1
# python infer_robovox.py --config ./configs/ECAPATDNN512_AAM2.yaml --epoch 71 --test_trial single --gpu_id 1 --eval_type easy_sn --submitted
# python infer_voxceleb.py --config ./configs/BC_Res2Net_AAM.yaml --epoch 58 --eval_type save_te --test_trial H --gpu_id 1 --enroll_frames 1000 --test_frames 200 --crop_option C
# python infer_voxceleb.py --config ./configs/ECAPATDNN_Light_AAM.yaml --epoch 69 --test_trial E --gpu_id 0 
# python infer_voxceleb.py --config ./configs/ECAPATDNN_Light_AAM.yaml --epoch 69 --eval_type save_te --test_trial H --gpu_id 0 --enroll_frames 1000 --test_frames 200 --crop_option C
# python infer_voices.py --config ./configs/BC_Res2Net_AAM.yaml --epoch 58 --eval_type save_te --test_trial dev --gpu_id 0 --enroll_frames 1000 --test_frames 200 --crop_option C
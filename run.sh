#!/bin/bash

# python train_voxceleb.py \
# --config ./configs/BC_Res2Net_AAM.yaml \

python DEV_train_voxceleb.py \
--config ./configs/ECAPATDNN512_AAM2.yaml \

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --distributed

# python train_voxceleb.py \
# --config ./configs/ResNetSE34L_AAM.yaml

# python infer_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --epoch 75 \
# --evaluate_type easy_sn\
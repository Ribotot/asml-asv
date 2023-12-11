#!/bin/bash

python train_voxceleb.py \
--config ./configs/ECAPATDNN512_AAM.yaml \
--distributed

# python train_voxceleb.py \
# --config ./configs/ECAPATDNN512_AAM.yaml \
# --distributed

# python train_voxceleb.py \
# --save_path exp/ResNetSE34L_AM_exp \
# --config ./configs/ResNetSE34L_AM.yaml
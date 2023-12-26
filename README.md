# ASML ASV

This repository contains the automatic speaker verification (ASV) training/inference code from the Acoustic, Speech Signal Processing, and Machine Learning (ASML) Lab. of Hanyang University.

Our work is based on VoxCeleb trainer from NAVER corpus, please refer to [LICENSE_clovaai.md](/LICENSE_clovaai.md) for details.

Code for other datasets other than VoxCeleb will be added. 

### model & loss

model and loss are organized into three folders: asml/clova/custom.

* The "asml" is our own work and the "clova" is the same as the those in the VoxCeleb trainer.

* "custom" contains what we are developing based on other researchers' work. 

### Please modify the codes before the run.
Please modify the data paths of following files

[train_voxceleb.py](/train_voxceleb.py) 

[infer_voxceleb.py](/infer_voxceleb.py) 

Please check the "dictkeys" and "speaker_label" of train_dataset_loader in [dataloader_voxceleb.py](/dataloader_voxceleb.py).

### License
```
Copyright (c) 2023-present ASML Lab. of Hanyang University,
All rights reserved.
```

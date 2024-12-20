# Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification (_ICLR 2023_)  [pdf](https://openreview.net/pdf?id=01KmhBsEPFO)

Authors: Jinxi Xiang, Xiyue Wang, Jun Zhang, Sen Yang, Xiao Han and Wei Yang. 

In this study, we investigate the low-rank property of whole slide images to establish a novel multiple-instance learning paradigm. Specifically, we enhance performance through a two-stage process:

1. We introduce a low-rank contrastive learning approach designed to generate pathology-specific visual representations.

2. We augment the standard transformer model by incorporating a learnable low-rank matrix, which serves as a surrogate to facilitate higher-order interactions.

## Requirements

Create a new enviroment with anaconda.
```shell
conda create -n ilra python=3.10 -y --no-default-packages
conda activate ilra
pip install --upgrade pip
pip install -r requirements.txt
```
## Structure

The folder structure is as follows:


```python

configs/    # create yaml file that contains dataloader, model, etc.
    ├── config_abmil_camelyon16_imagenet.yaml
    ├── config_clam_camelyon16_imagenet.yaml
    └── ...
models/  # definition of MIL models
    ├── abmil.py
    ├── clam.py
    ├── ilra.py   # our proposed model
    └── ...
splits/  # training ans test data split
    ├── camelyon16_test.csv
    ├── camelyon16_train_10fold.csv
    └── ...
topk/ # dependency of CLAM
    └── ...
train_mil.py/   # main function of training and test
wsi_dataset.py/  # WSI dataset
```

### Step 1
Prepare your WSI feature with [CLAM](https://github.com/mahmoodlab/CLAM) and change the 'Data.feat_dir' in yaml file. 
To run this example code, you can download the CAMELYON16 features using ImageNet pre-trained Resnet50 from this [link](https://drive.google.com/file/d/1fJ_weyjPcpLEEVpQPwjFnZCWy_47VcRY/view?usp=sharing).

### Step 2
change _line 42_ in __train_mil.py__ to run different experiments.
```python
fname = "config_ilra_camelyon16_imagenet"
```

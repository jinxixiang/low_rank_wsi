# Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification (_ICLR 2023 poster_)  [pdf](https://openreview.net/pdf?id=01KmhBsEPFO)

Author: Jinxi Xiang, Xiyue Wang, Jun Zhang, Sen Yang, Xiao Han, Wei Yang. 

Affiliation: Tencent AI Lab.


We explore the low-rank property of whole slide images to develop a new multiple-instance learning paradigm. In concrete, we improve the performance in two stages:

1. We propose low-rank contrastive learning for pathology-specific visual representation.

2. We improve the vanilla transformer by adding a learnable low-rank matrix as a surrogate to implement higher-order interaction.

## Requirements

use the _environment.yaml_ with conda.

## Structure

The folder structure is as follows:


```python
mil/
    └──configs/    # create yaml file that contains dataloader, model, etc.
        ├── config_abmil_camelyon16_imagenet.yaml
        ├── config_clam_camelyon16_imagenet.yaml
        └── ...
    └──models/  # definition of MIL models
        ├── abmil.py
        ├── clam.py
        ├── ilra.py   # our proposed model
        └── ...
    └──splits/  # training ans test data split
        ├── camelyon16_test.csv
        ├── camelyon16_train_10fold.csv
        └── ...
    └──topk/ # dependency of CLAM
        └── ...
    └──train_mil.py/   # main function of training and test
    └──wsi_dataset.py/  # WSI dataset
```

### Step 1
Prepare your WSI feature with [CLAM](https://github.com/mahmoodlab/CLAM) and change the 'Data.feat_dir' in yaml file. 
To run this example code, you can download the CAMELYON16 features using ImageNet pre-trained Resnet50 from this [link](https://drive.google.com/file/d/1fJ_weyjPcpLEEVpQPwjFnZCWy_47VcRY/view?usp=sharing).

### Step 2
change _line 26_ in __train_mil.py__ to run different experiments.

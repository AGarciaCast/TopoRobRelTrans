# Relative Representations: Topological and Geometric Perspectives
This repository contains the code for the paper called **"Relative Representations: Topological and Geometric Perspectives"**.

## Abstract
Relative representations are an established approach to zero-shot model stitching, consisting of a non-trainable transformation of the latent space of a deep neural network. Based on insights of topological and geometric nature, we propose two improvements to relative representations. First, we introduce a normalization procedure in the relative transformation, resulting in invariance to non-isotropic rescalings and permutations. The latter coincides with the symmetries in parameter space induced by common activation functions. Second, we propose to deploy topological densification when fine-tuning relative representations, a topological regularization loss encouraging clustering within classes. We provide an empirical investigation on a natural language task, where both the proposed variations yield improved performance on zero-shot model stitching. 

## Information
- The code for the method is in `modules` and `pl_modules`. <br />
- The experiments can be reproduced using the notebook `topo_multilingual_stitching.ipynb` and the script `train.py`. <br />

## Requirements

To install the requirements, we use conda. We recommend creating a new environment for the project.
```
conda create -n topo-rel python=3.7.12
conda activate topo-rel
```

Install the relevant dependencies.
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```






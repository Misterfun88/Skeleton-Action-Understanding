# Unified Multi-modal Unsupervised Representation Learning for Skeleton-based Action Understanding

**This is a repository containing the implementation of an ACM MM 2023 paper.**

**Paper Link**: [arXiv](https://arxiv.org/abs/2311.03106), [ACM DL](https://dl.acm.org/doi/10.1145/3581783.3612449)

## Requirements

Instructions for setting up the conda environment. 
```
conda create -n umurl python=3.9 anaconda
conda activate umurl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
pip3 install tensorboard
```

## Data Preparation

Details on the raw data and preprocessing data can be found within the data and data_gen directories respectively.

## Pretraining and Evaluation

Detailed information on pre-training and evaluating on the downstream task is within the repository.

## Pretrained Models

We have released several pre-trained models: [Google Drive](https://drive.google.com/drive/folders/1vDGfEFRVDEU5VnutrHmyAb9RnZT_udF4?usp=sharing)

## Visualization

Examples of t-SNE visualizations are provided to show the effectiveness of our proposed model.

## Citation

If you find this repository useful, please consider citing our paper:

## Acknowledgements

This work was supported by numerous grant
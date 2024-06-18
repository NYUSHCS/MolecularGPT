# MolecularGPT
### This is the code for paper MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction

# GraphFM
Official code for "[GraphFM: A Comprehensive Benchmark for Graph Foundation Model](https://arxiv.org/abs/2406.08310v2)". GraphFM is a comprehensive benchmark for Graph Foundation Models (Graph FMs) and is based on graph self-supervised learning (GSSL). It aims to research the homogenization and scalability of Graph FMs. 

## Overview of the Benchmark
GraphFM provides a fair and comprehensive platform to evaluate existing GSSL works and facilitate future research.

![architecture](https://github.com/NYUSHCS/GraphFM/blob/main/img/)

We perform a comprehensive benchmark of state-of-the-art self-supervised GNN models through four key aspects: dataset scale, training strategies, GSSL methods for Graph FMs, and adaptability to different downstream tasks.

## Installation
The required packages can be installed by running `pip install -r requirements.txt`.

## 🚀Quick Start

### Set up Model, Dataset and Batch type parameters

**model** ： 
`BGRL`, `CCA-SSG`, `GBT`, `GCA`, `GraphECL`, `GraphMAE`, `GraphMAE2`, `S2GAE`

**dataset** ： 
`cora`, `pubmed`, `citeseer`, `Flickr`, `Reddit`, `ogbn-arxiv`

**batch_type** ：
`full_batch`, `node_sampling`, `subgraph_sampling`

### Get Best Hyperparameters
You can run the `python main_optuna.py --type_model $model --dataset $dataset --batch_type $batch_type` to get the best hyperparameters.
#### Test the performance on classification tasks 
CUDA_VISIBLE_DEVICES=0 python downstream_test_llama_cla.py \
    --load_8bit \
    --base_model /home/leslie/Llama-2-7b-chat-hf \
### Train the Main Code
You can train the model with `main.py` after obtaining the hyperparameters tuned by Optuna.

## 📱️Updates
2024.6.15 Submitted our paper to arXiv.

## Reference Code

| **ID** | **Project** | 
|--------|---------|
| 1      | [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514)      | 
| 2      | [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) | 

## Citation

If you find this repo useful, please star the repo and cite:

```bibtex
@article{xu2024graphfm,
      title={GraphFM: A Comprehensive Benchmark for Graph Foundation Model},
      author={Xu, Yuhao and Liu, Xinqi and Duan, Keyu and Fang, Yi and Chuang, Yu-Neng and Zha, Daochen and Tan, Qiaoyu},
      journal={arXiv preprint arXiv:2406.08310},
      year={2024}
    }
``

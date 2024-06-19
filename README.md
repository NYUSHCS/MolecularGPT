# MolecularGPT
Official code for "[MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction]".

## Installation
The required packages can be installed by running 
`pip install -r requirements.txt`.

## 🚀Quick Start

### Download the Datasets
....
### Construct the K-Shot instruction datasets
...
### Train the model
...
### Evaluate the performance on classification tasks 
`python downstream_test_llama_cla.py --load_8bit --base_model $model --lora_weights $lora --path $path --shot $shot` 
### Evaluate the performance on regression tasks 
`python downstream_test_llama_reg.py --load_8bit --base_model $model --lora_weights $lora --path $path --shot $shot` 
### Evaluate the bacelines
...

## 📱️Updates
2024.6.19 Submitted our paper to arXiv.

## Reference Code

| **ID** | **Project** | 
|--------|---------|
| 1      | [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514)      | 
| 2      | [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) | 

## Citation

If you find this repo useful, please star the repo and cite:

```bibtex
@article{MolecularGPT,
      title={MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction},
      year={2024}
    }
``

# MolecularGPT
Official code for "[MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction]".

## 🚀Quick Start
### Installation
The required packages can be installed by running 
`conda create -n MolecularGPT python==3.10` \
`conda activate MolecularGPT` \
`cd .\MolecularGPT` \
`bash init_env.sh` \
### Download the Datasets
**MoleculeNet Datasts**
`wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip`\
`unzip chem_dataset.zip`\
`mv dataset property_data`\
**CYP450 Datasts**
Downloaded from https://github.com/shenwanxiang/ChemBench/blob/master/src/chembench/data_and_index/CYP450/CYP450.csv.gz\
Then uncompress file CYP450.csv to ./property_data/cyp450/raw/CYP450.csv.\
### Construct the K-Shot instruction datasets

### Train the model
#### Download LLaMA2-7b-chat from huggingface
Download from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and move to `./ckpts/llama`\
#### Train the MolecularGPT

### Evaluate the model
#### Download LoRA Weighs form huggingface
Download the `adapter_config.json` and `adapter_model.bin` from https://huggingface.co/YuyanLiu/MolecularGPT and move to `./ckpts/lora`\
#### Evaluate the performance on classification tasks 
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
| 1      | [GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot Learning](https://github.com/zhao-ht/GIMLET)      | 
| 2      | [Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models](https://github.com/zjunlp/Mol-Instructions) | 
| 3      | [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) |
| 4      | [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) |
| 5      | [usearch-molecules](https://github.com/ashvardanian/usearch-molecules) |


## Citation

If you find this repo useful, please star the repo and cite:

```bibtex
@article{MolecularGPT,
      title={MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction},
      year={2024}
    }
``

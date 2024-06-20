# MolecularGPT
Official code for "[MolecularGPT: Open Large Language Model (LLM) for Few-Shot Molecular Property Prediction]".

## 🚀Quick Start
### Installation
The required packages can be installed by running
```
conda create -n MolecularGPT python==3.10
conda activate MolecularGPT
cd .\MolecularGPT 
bash init_env.sh 
pip install git+https://github.com/ashvardanian/usearch-molecules.git@main
```
### Download the Datasets
#### Train datasets

**Chembl dataset**
```
cd prompt_data/ 
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip 
unzip dataPythonReduced.zip 

cd dataPythonReduced 
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl 
wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl 

cd .. 
rm dataPythonReduced.zip 
mkdir -p chembl_raw/raw 
mv dataPythonReduced/* chembl_raw/raw 
wget 'https://www.dropbox.com/s/vi084g0ol06wzkt/mol_cluster.csv?dl=1' 
mv 'mol_cluster.csv?dl=1' chembl_raw/raw/mol_cluster.csv

python transform.py --input-dir chembl_raw/raw --output-dir chembl_full > transform.out 
cd .. 
```
**Chembl property dataset**
```
cd prompt_data
filename='mole_graph_property.csv'
fileid='1oLxIDOzp8MY0Jhzc1m6E7SCOVAZO5L4D'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
cd ..
```
**QM9 dataset**
...

#### Test datasets
**MoleculeNet Datasts** 
```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset property_data
```
**CYP450 Datasts** \
Downloaded from https://github.com/shenwanxiang/ChemBench/blob/master/src/chembench/data_and_index/CYP450/CYP450.csv.gz \
Then uncompress file CYP450.csv to ./property_data/cyp450/raw/CYP450.csv.
### Construct the K-Shot instruction datasets
#### Train datasets
```
cd prompts
python generate_pretrain_dataset.py --generate_assay_text --generate_mole_text --generate_qm9_text --split_non_overlap --add_negation

python prep_encode_train.py
python prep_index_train.py
python ICL_train.py
```

#### Test datasets
```
python prep_test_dataset_aug.py --prompt_augmentation ''
python prep_encode_test.py
python prep_index_test.py
```
For classfication and regression task:
```
ICL_test_sim_cls.py
ICL_test_sim_reg.py
```
To construct the k-shot instructions arranged by ascending order: 
```
ICL_test_reverse_cls.py
ICL_test_reverse_reg.py
```
To construct the k-shot instructions retrieved based on diversity : 
```
ICL_test_diversity.py
```

### Train the model
#### Download LLaMA2-7b-chat from huggingface
Download from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and move to `./ckpts/llama`
#### Train the MolecularGPT

### Evaluate the model
#### Download LoRA Weighs form huggingface
Download the `adapter_config.json` and `adapter_model.bin` from https://huggingface.co/YuyanLiu/MolecularGPT and move to `./ckpts/lora`
#### Evaluate the performance on classification tasks 
```
python downstream_test_llama_cla.py --load_8bit --base_model $model --lora_weights $lora --path $path --shot $shot
```
### Evaluate the performance on regression tasks 
```
python downstream_test_llama_reg.py --load_8bit --base_model $model --lora_weights $lora --path $path --shot $shot
``` 
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

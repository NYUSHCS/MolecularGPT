from rdkit import Chem
import numpy as np
import tqdm
import datetime 
import pyarrow as pa
import pandas as pd
from fastparquet import write
import os
import pyarrow.parquet as pq
from dataset import FingerprintedDataset, shape_maccs
from pathlib import Path


def create_prompt(results_smiles, results_label):
    prompt=''
    for i in range(len(results_smiles)):
        prompt += f"SMILES: {results_smiles[i]}\nlabel: {results_label[i]}\n"

    return prompt

datas = ['assay','property','qm9']

for data in datas:
    path = './train_process/'
    path  = os.path.join(path,data)
    file_names = [f.name for f in Path(path).iterdir() if f.is_dir()]
    for file_name in file_names:
        dir  = os.path.join(path, file_name)
        print(dir)
        data = FingerprintedDataset.open(dir, shape=shape_maccs)
        filenames = sorted(os.listdir(dir))
        for filename in filenames:
            if not filename.endswith(".parquet"):
                continue
            table = pq.read_table(os.path.join(dir, filename))
            maccs = table['maccs']
            smiles = table['graph']
            label = table['label']
            text = table['text']
            assay = table['assayid'][0]
            
            # sentences = str(table['text'][0]).split(". ")
            # front_sentences = ". ".join(sentences[:-1])
            # last_sentence = sentences[-1]
            
            data_1shot = []
            data_2shot = []
            data_3shot = []
            data_4shot = []

              
            i=0
            
            macc = list(macc.as_py())
            for macc in maccs:

                results =data.search(macc, 5)
                index = [r[0] for r in results]
                # print(datetime.datetime.now())
                results_keys = [r[0] for r in results]
                results_smiles = [r[1] for r in results]
                results_scores = [r[2] for r in results]
                results_label = [r[3] for r in results]
                
                
                sentences = str(text[i]).split(". ")
                front_sentences = ". ".join(sentences[:-1])
                last_sentence = sentences[-1]

                
                # for property dataset, its instruction will have one sentence
                if len(front_sentences)==0:
                   
                    # for train dataset
                    data_1shot.append([f'Here are some examples about molecule {assay}.\n' + create_prompt(results_smiles[1], results_label[1]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_2shot.append([f'Here are some examples about molecule {assay}.\n' + create_prompt(results_smiles[1:3], results_label[1:3]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_3shot.append([f'Here are some examples about molecule {assay}.\n' + create_prompt(results_smiles[1:4], results_label[1:4]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_4shot.append([f'Here are some examples about molecule {assay}.\n' + create_prompt(results_smiles[1:], results_label[1:]) + last_sentence, str(smiles[i]), str(label[i])])

                    
                else:

                    # for train dataset
                    data_1shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[1], results_label[1]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_2shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[1:3], results_label[1:3]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_3shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[1:4], results_label[1:4]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_4shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[1:], results_label[1:]) + last_sentence, str(smiles[i]), str(label[i])])

                
                i = i+1
            
            new_columns_1shot = pd.DataFrame(data_1shot, columns=['instruction', 'input', 'output'])
            new_columns_2shot = pd.DataFrame(data_2shot, columns=['instruction', 'input', 'output'])
            new_columns_3shot = pd.DataFrame(data_3shot, columns=['instruction', 'input', 'output'])
            new_columns_4shot = pd.DataFrame(data_4shot, columns=['instruction', 'input', 'output'])

            
            new_path = './train_dataset/1-shot'
            os.makedirs(os.path.join(new_path, file_name), exist_ok=True)
            new_columns_1shot.to_json(os.path.join(new_path, file_name, + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, file_name, + '.json'))
            
            new_path = './train_dataset/2-shot'
            os.makedirs(os.path.join(new_path, file_name), exist_ok=True)
            new_columns_2shot.to_json(os.path.join(new_path, file_name, + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, file_name, + '.json'))
                        
            new_path = './train_dataset/3-shot'
            os.makedirs(os.path.join(new_path, file_name), exist_ok=True)
            new_columns_3shot.to_json(os.path.join(new_path, file_name, + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, file_name, + '.json'))
                        
            new_path = './train_dataset/4-shot'
            os.makedirs(os.path.join(new_path, file_name), exist_ok=True)            
            new_columns_4shot.to_json(os.path.join(new_path, file_name, + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, file_name, + '.json'))
            
print('saved !')
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



def create_prompt(results_smiles, results_label, shot):
    prompt = ''
    prompt_yes=''
    prompt_no=''
    num_yes = 0
    num_no = 0
    for i in range(len(results_smiles)):
                
        # classification positive and negtive tasks
        if num_yes < shot and results_label[i] == str(1.0):
            prompt_yes += f"SMILES: {results_smiles[i]}\nlabel: Yes\n"
            num_yes += 1
        if num_no < shot and results_label[i] == str(0.0):
            prompt_no += f"SMILES: {results_smiles[i]}\nlabel: No\n"
            num_no += 1
    prompt = prompt_yes + prompt_no

    return prompt

datasets = [
    'bace',
    'bbbp',
    'cyp450',
    'hiv',
    'muv',
    'tox21',
    'toxcast',
            ]


for dataset in datasets:
    path = './test_process'
    path = os.path.join(path, dataset)

    for file in os.listdir(path):
        if file.startswith("test") and not file.endswith("_add.parquet"):
            num = file.split('_')[-1].split('.')[0]
            table_path = os.path.join(path, file)
            data_path = os.path.join(path, num)

            data = FingerprintedDataset.open(data_path, shape=shape_maccs)

            table = pq.read_table(table_path)
            # print(f'table: {len(table)}')
            maccs = table['maccs']
            smiles = table['smiles']
            label = table['label']
            
            ins = pd.read_json(os.path.join('./test_dataset/0-shot/', dataset, num+'.json'), lines=True)
            text = ins['instruction']

            

            data_2shot = []

            data_4shot = []

            data_6shot = []

            data_8shot = []


            i=0

            for macc in maccs:
                macc = list(macc.as_py())
                results =data.search(macc, 1000)  # top 1000 molecules

                index = [r[0] for r in results]

                results_keys = [r[0] for r in results]
                results_smiles = [r[1] for r in results]
                results_scores = [r[2] for r in results]
                results_label = [r[3] for r in results]
    
                
                sentences = str(text[i]).split(". ")
                front_sentences = ". ".join(sentences[:-1])
                last_sentence = sentences[-1]
                

                
                # for property dataset, its instruction will have one sentence
                if len(front_sentences)==0:
                    # for test dataset
                    
                    data_2shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles, results_label, 1) + last_sentence, str(smiles[i]), str(label[i])])
                   
                    data_4shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles, results_label, 2) + last_sentence, str(smiles[i]), str(label[i])])

                    data_6shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles, results_label, 3) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    data_8shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles, results_label, 4) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    
                else:

                    # for test dataset

                    data_2shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles, results_label, 1) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    data_4shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles, results_label, 2) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    data_6shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles, results_label, 3) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    data_8shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles, results_label, 4) + last_sentence, str(smiles[i]), str(label[i])])

                i = i+1

            
            
            new_columns_2shot = pd.DataFrame(data_2shot, columns=['instruction', 'input', 'output'])

            new_columns_4shot = pd.DataFrame(data_4shot, columns=['instruction', 'input', 'output'])

            new_columns_6shot = pd.DataFrame(data_6shot, columns=['instruction', 'input', 'output'])

            new_columns_8shot = pd.DataFrame(data_8shot, columns=['instruction', 'input', 'output'])
            
            
            
            new_path = './test_dataset/2-shot-class'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_2shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        
                        
            new_path = './test_dataset/4-shot-class'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)            
            new_columns_4shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        
                        
            new_path = './test_dataset/6-shot-class/'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_6shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        
            new_path = './test_dataset/8-shot-class/'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)            
            new_columns_8shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))

print('saved !')
            
            


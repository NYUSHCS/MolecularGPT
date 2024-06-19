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
        
        # classifaction tasks
        if results_label[i] == str(1.0):
            prompt += f"SMILES: {results_smiles[i]}\nlabel: Yes\n"
        elif results_label[i] == str(0.0):
            prompt += f"SMILES: {results_smiles[i]}\nlabel: No\n"
        else:
            print(f'label is not 1.0 or 0.0')
            
            
        # regression tasks
        # prompt += f"SMILES: {results_smiles[i]}\nlabel: {results_label[i]}\n"

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

#regression tasks:
# datasets = ['esol','freesolv','lipo']



for dataset in datasets:
    path = './test_process'
    path = os.path.join(path, dataset)
    # files = [f.name for f in Path(path).iterdir() if f.is_dir()]
    for file in os.listdir(path):
        if file.startswith("test") and not file.endswith("_add.parquet"):
            num = file.split('_')[-1].split('.')[0]
            table_path = os.path.join(path, file)
            data_path = os.path.join(path, num)

            # data = FingerprintedDataset.open(dir, shape=shape_maccs)
            data = FingerprintedDataset.open(data_path, shape=shape_maccs)

            table = pq.read_table(table_path)
            print(f'table: {len(table)}')
            maccs = table['maccs']
            smiles = table['smiles']
            label = table['label']
            
            ins = pd.read_json(os.path.join('./test_dataset/0-shot/', dataset, num+'.json'), lines=True)
            text = ins['instruction']
            # text = table['instruction']
            # assay = table['assayid'][0]
            
            data_1shot = []
            data_2shot = []
            data_3shot = []
            data_4shot = []
            data_5shot = []
            data_6shot = []
            data_7shot = []
            data_8shot = []
              
            i=0

            for macc in maccs:
                macc = list(macc.as_py())
                results =data.search(macc, 8)

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
                    data_1shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:1], results_label[:1]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_2shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:2], results_label[:2]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_3shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:3], results_label[:3]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_4shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:4], results_label[:4]) + last_sentence, str(smiles[i]), str(label[i])])
                    # data_5shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:5], results_label[:5]) + last_sentence, smiles[i], label[i]])
                    data_6shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:6], results_label[:6]) + last_sentence, str(smiles[i]), str(label[i])])
                    # data_7shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles[:7], results_label[:7]) + last_sentence, smiles[i], label[i]])
                    data_8shot.append([f'Here are some examples about molecular property.\n' + create_prompt(results_smiles, results_label) + last_sentence, str(smiles[i]), str(label[i])])
                    
                    
                else:

                    # for test dataset
                    data_1shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:1], results_label[:1]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_2shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:2], results_label[:2]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_3shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:3], results_label[:3]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_4shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:4], results_label[:4]) + last_sentence, str(smiles[i]), str(label[i])])
                    # data_5shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:5], results_label[:5]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_6shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:6], results_label[:6]) + last_sentence, str(smiles[i]), str(label[i])])
                    # data_7shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles[:7], results_label[:7]) + last_sentence, str(smiles[i]), str(label[i])])
                    data_8shot.append([front_sentences +'. Here are some examples.\n' + create_prompt(results_smiles, results_label) + last_sentence, str(smiles[i]), str(label[i])])  #float

                
                i = i+1
            
            new_columns_1shot = pd.DataFrame(data_1shot, columns=['instruction', 'input', 'output'])
            new_columns_2shot = pd.DataFrame(data_2shot, columns=['instruction', 'input', 'output'])
            new_columns_3shot = pd.DataFrame(data_3shot, columns=['instruction', 'input', 'output'])
            new_columns_4shot = pd.DataFrame(data_4shot, columns=['instruction', 'input', 'output'])
            # new_columns_5shot = pd.DataFrame(data_5shot, columns=['instruction', 'input', 'output'])
            new_columns_6shot = pd.DataFrame(data_6shot, columns=['instruction', 'input', 'output'])
            # new_columns_7shot = pd.DataFrame(data_7shot, columns=['instruction', 'input', 'output'])
            new_columns_8shot = pd.DataFrame(data_8shot, columns=['instruction', 'input', 'output'])
            
            
            new_path = './test_dataset/1-shot'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_1shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
            
            new_path = './test_dataset/2-shot'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_2shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        
            new_path = './test_dataset/3-shot'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_3shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        
            new_path = './test_dataset/4-shot'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)            
            new_columns_4shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        

            new_path = './test_dataset/6-shot/'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)
            new_columns_6shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))
                        

            new_path = './test_dataset/8-shot/'
            os.makedirs(os.path.join(new_path, dataset), exist_ok=True)            
            new_columns_8shot.to_json(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'), orient='records', lines=True)
            print(os.path.join(new_path, dataset, file.split('_')[1] + '_' + num + '.json'))

print('saved !')
            
            


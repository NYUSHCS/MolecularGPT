from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.metrics import (r2_score,
                             roc_auc_score)
from dataloaders.splitters import scaffold_split
# from dataloaders.splitters import random_scaffold_split, random_split, scaffold_split
from torch.utils.data import DataLoader

# import plotly.graph_objects as go
import argparse

import commentjson
# from basic_pipeline import load_graph_args,eval_result
# from model import get_model
from dataloaders import add_prompt_transform_dict,\
    graph_text_collator_dict, \
    MoleculeDatasetSplitLabel,graph_text_tokenizer_dict

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm
import os
import re
# from datasets import load_dataset
import pandas as pd
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

# about seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rich_features',action='store_true')
parser.add_argument('--single_split',type=int,default=None)
parser.add_argument('--task_policy',type=str,default='traversal', choices=['single','traversal','multi_mixture','multi_label'])
parser.add_argument('--split', type=str, default='scaffold')

# set instruction type!!!
parser.add_argument('--prompt_augmentation',default='',choices=['','rewrite','expand','detail','shorten','name'])

parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--no_cuda',action='store_true')
# parser.add_argument('--prompt_file',type=str,default='selected_prompt_downstream_task.json')
parser.add_argument('--prompt_file',type=str,default='augmented_selected_prompt_downstream_task.json')

args,left = parser.parse_known_args()
print('arguments\t', args)
args.split_label=args.task_policy=='multi'

def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor','esol','freesolv','lipo']:
        return 1
    elif dataset == 'pcba':
        return 128
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'cyp450':
        return 5
    raise ValueError(dataset + ': Invalid dataset name.')

def task_type(dataset):
    if dataset in ['esol','freesolv','lipo']:
        return 'reg'
    else:
        return 'cla'

def better_result(result,reference,dataset):
    if task_type(dataset)=='cla':
        return result>reference
    else:
        assert task_type(dataset)=='reg'
        return result<reference



if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if (torch.cuda.is_available() and not args.no_cuda) else torch.device('cpu')
    if torch.cuda.is_available() and not args.no_cuda :
        torch.cuda.manual_seed_all(args.runseed)

    #Load instruction files, and add them for molecule data.

    # classfication and regression tasks
    datasets = ['bace', 'bbbp' , 'hiv', 'toxcast', 'tox21',  'cyp450', 'muv', 'esol', 'freesolv', 'lipo']

    for data in datasets:
        if args.prompt_augmentation=='':
            with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                prompts = commentjson.load(load_f)
            prompt=prompts[data]
        else:
            with open(os.path.join("prompts",args.prompt_file), 'r') as load_f:
                prompts = commentjson.load(load_f)
            prompt_all=prompts[data]
            prompt={}
            for key in prompt_all:
                if args.prompt_augmentation in prompt_all[key]:
                    prompt[key]=prompt_all[key][args.prompt_augmentation]
                else:
                    print('label split {} has no augmentation {}'.format(key, args.prompt_augmentation))
        
        
        # Bunch of classification tasks
        # class tasks      
            
        num_tasks = get_num_task(data)  
        recurrent_range=range(num_tasks)
        
        
        # for single_split_label in recurrent_range:
        dataset_folder = './property_data/'
        
        dataset = MoleculeDatasetSplitLabel(root=dataset_folder, name=data,return_smiles=True,split_label=args.split_label,single_split=args.single_split,rich_features=args.rich_features)
        if args.split == 'scaffold':
            # if args.single_split is not None:
            
            smiles_list = pd.read_csv(dataset_folder + data + '/processed/smiles.csv',
                                        header=None)[0].tolist()
            train_index, valid_index, test_index = scaffold_split(
                torch.arange(len(smiles_list)), smiles_list, null_value=0, frac_train=0.8,
                frac_valid=0.1, frac_test=0.1)

            train_index_total=[]
            valid_index_total=[]
            test_index_total=[]
            for times in range(dataset.label_number):
                train_index_times=train_index+times*dataset.len_oridata()
                valid_index_times = valid_index + times * dataset.len_oridata()
                test_index_times = test_index + times * dataset.len_oridata()

                train_index_total.append(train_index_times)
                valid_index_total.append(valid_index_times)
                test_index_total.append(test_index_times)
            train_index_total=torch.cat(train_index_total,0)
            valid_index_total=torch.cat(valid_index_total,0)
            test_index_total=torch.cat(test_index_total,0)

            train_dataset = dataset[train_index_total]
            valid_dataset = dataset[valid_index_total]
            test_dataset = dataset[test_index_total]
            
        labels = int(num_tasks)
            
        # test dataset
        for label_id in args.prompt_id.keys():
            prompt_list =[]
            smiles_list =[]
            label_list = []
            for i in range(len(test_dataset)):
                label = test_dataset[i].y[0, int(label_id)].item()

                if label == -100.0 or label == -100:
                    continue
                
                if task_type(data)=='cla':
                    if label == 1.0 or label == 1:
                        label_list.append('Yes')  
                    if label == 0.0 or label == 0:
                        label_list.append('No')
                        
                if task_type(data)=='reg':
                    label_list.append(label) 
         
                prompt_list.append(prompt[label_id][0])

                smiles_list.append(test_dataset[i].smiles)
            
            
            df = pd.DataFrame({
            'instruction': prompt_list,
            'smiles': smiles_list,
            'label': label_list
            })
        

            # to parquet
            new_path = os.path.join('./test_process/', data)
            os.makedirs(new_path, exist_ok=True)
            df.to_parquet(os.path.join(new_path, 'test_{}_{}.parquet'.format(data, label_id)), index=False)
            
            # to zero shot json
            
            df = df.rename(columns={
            'smiles': 'input',
            'label': 'output'
        })

            # to zero shot json
            if args.prompt_augmentation=='':
                os.makedirs(os.path.join('./test_dataset/0-shot', data), exist_ok=True)
                df.to_json(os.path.join('./test_dataset/0-shot', data) + '/{}.json'.format(int(label_id)), orient='records', lines=True)
            else:   
                os.makedirs(os.path.join('./test_dataset/0-shot-'+ args.prompt_augmentation, data), exist_ok=True)
                df.to_json(os.path.join('./test_dataset/0-shot-'+ args.prompt_augmentation, data) + '/{}.json'.format(int(label_id)), orient='records', lines=True)
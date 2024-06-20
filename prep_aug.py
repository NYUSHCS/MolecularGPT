import commentjson
import os
import pandas as pd
path = './prompts/augmented_selected_prompt_downstream_task.json'

with open(path, 'r') as load_f:
    prompts_selected = commentjson.load(load_f)

datasets = [
    'bace',
    'bbbp',
    'cyp450',
    'hiv',
    'muv',
    'tox21',
    'toxcast',
            ]

augs = [
    "rewrite", 
    "expand", 
    "detail", 
    "shorten",
        ]

path_new = './test_dataset/0-shot-'

for dataset in datasets:
    path1 = './test_dataset/0-shot/'
    path1 = os.path.join(path1, dataset)
    for f in os.listdir(path1):
        
        print(os.path.join(path1, f))
        df = pd.read_json(os.path.join(path1, f), lines=True)
        name = f.split(".")[0]
        for aug in augs:
            try:
                sentences = prompts_selected[dataset][name][aug]
                # sentences = ["Only output Yes or No!!! "]+sentences
                sentences = ["".join(sentences)]
                # print(sentences)
                df['instruction'] = sentences * len(df)
                path2 = os.path.join(path_new + aug, dataset)
                os.makedirs(path2, exist_ok=True)
                
                df.to_json(os.path.join(path2, f), orient='records', lines=True)
            except:
                continue

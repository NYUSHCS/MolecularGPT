import pyarrow.parquet as pq
import pandas as pd
import os
import random


path0 = './train_dataset/0-shot'
path1 = './train_dataset/1-shot'
path2 = './train_dataset/2-shot'
path3 = './train_dataset/3-shot'
path4 = './train_dataset/4-shot'
path = './train_dataset/01234-shot'

for f in os.listdir(path0):
    
    df0 = pd.read_json(os.path.join(path0, f), lines=True)
    df1 = pd.read_json(os.path.join(path1, f), lines=True)
    df2 = pd.read_json(os.path.join(path2, f), lines=True)
    df3 = pd.read_json(os.path.join(path3, f), lines=True)
    df4 = pd.read_json(os.path.join(path4, f), lines=True)
    
    lst = list(df0.index)


    # 划分大小
    sizes = [int(len(lst)*0.6), int(len(lst)*0.1), int(len(lst)*0.1), int(len(lst)*0.1), int(len(lst)*0.1)]

    # 用于存储每个子列表的列表
    sublists = []

    # 对每个大小进行处理
    for size in sizes:
        # 随机抽取一部分数据
        subset = random.sample(lst, size)
        
        # 将抽取的数据从原始列表中移除，以避免重复
        lst = [x for x in lst if x not in subset]
        
        # 将子列表添加到列表中
        sublists.append(subset)

    sub0 = sublists[0]
    sub1 = sublists[1]
    sub2 = sublists[2]
    sub3 = sublists[3]
    sub4 = sublists[4]
    
    
    half0 = df0.loc[sub0]
    half1 = df1.loc[sub1]
    half2 = df2.loc[sub2]
    half3 = df3.loc[sub3]
    half4 = df4.loc[sub4]

    result = pd.concat([half0, half1, half2, half3, half4])
    result.to_json(os.path.join(path, f), orient='records', lines=True)
    print(os.path.join(path, f))
    
print(f'down!')
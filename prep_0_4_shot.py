import pyarrow.parquet as pq
import pandas as pd
import os
import random

# path1 = '/gpfsnyu/scratch/jd5849/data/data_0&4shot/'
# path2 = '/gpfsnyu/scratch/jd5849/data/my_data_4shot/'
# path3 = '/gpfsnyu/scratch/jd5849/data/my_data_0_4shot/'


# for f in os.listdir(path1):
#     path_0 = os.path.join(path1,f)
#     table = pq.read_table(path_0).to_pandas()
#     table  = pd.DataFrame(table, columns = ['text','graph', 'label'])
#     df1 = table.rename(columns={'text':'instruction', 'graph': 'input', 'label': 'output'})
#     df1.to_json(path3 + f.replace('.parquet','.json'), orient='records', lines=True)

    
#     # load
#     df1 = pd.read_json(os.path.join(path3, f.replace('.parquet','.json')), lines=True)
#     df2 = pd.read_json(os.path.join(path2, f.replace('.parquet','.json')), lines=True)
#     indices = list(df1.index)

#     # 随机选择一半的索引
#     selected_indices = random.sample(indices, len(indices) // 2)
#     # print(selected_indices)
#     # 从每个DataFrame中选择对应的元素
#     half1 = df1.loc[selected_indices]
#     half2 = df2.drop(selected_indices)

#     # 合并结果
#     result = pd.concat([half1, half2])

#     # 将结果写入一个新的JSON文件
#     result.to_json(path3 + f.replace('.parquet','.json'), orient='records', lines=True)
#     print(path3 + f.replace('.parquet','.json'))


path1 = './train_dataset/0-shot'
path2 = './train_dataset/4-shot'
path3 = './train_dataset/0-4-shot'

for f in os.listdir(path1):
    df1 = pd.read_json(os.path.join(path1, f), lines=True)
    df2 = pd.read_json(os.path.join(path2, f), lines=True)
    
    indices = list(df1.index)

    selected_indices = random.sample(indices, len(indices) // 2)
    half1 = df1.loc[selected_indices]
    half2 = df2.drop(selected_indices)
    result = pd.concat([half1, half2])
    result.to_json(os.path.join(path3, f), orient='records', lines=True)
    print(os.path.join(path3, f))
    
print(f'down!')
import os
import sys
import random
from glob import glob
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.cm.get_cmap('coolwarm')

import torch

import librosa

from config import CFG
from pytorch_modeler import set_seed
import stereo2mono_preprocessing as prep

set_seed(CFG.seed)

df_23 = pd.read_csv(f'{CFG.BASE_PATH3}/train_metadata.csv')
df_23['filepath'] = CFG.BASE_PATH3 + '/train_audio/' + df_23.filename
df_23['target'] = df_23.primary_label.map(CFG.name2label)
df_23['birdclef'] = '23'
df_23['filename'] = df_23.filepath.map(lambda x: x.split('/')[-1])
df_23['xc_id'] = df_23.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
#assert tf.io.gfile.exists(df_23.filepath.iloc[0])

# Display rwos
print("# Samples in BirdCLEF 23: {:,}".format(len(df_23)))
df_23.head(2).style.set_caption("BirdCLEF - 23").set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'blue'),
        ('font-size', '16px')
    ]
}])

# BirdCLEF-2020
df_20 = pd.read_csv(f'{CFG.BASE_PATH0}/train.csv')
df_20['primary_label'] = df_20['ebird_code']
df_20['filepath'] = CFG.BASE_PATH0 + '/train_audio/' + df_20.primary_label + '/' + df_20.filename
df_20['scientific_name'] = df_20['sci_name']
df_20['common_name'] = df_20['species']
df_20['target'] = df_20.primary_label.map(CFG.name2label2)
df_20['birdclef'] = '20'
#assert tf.io.gfile.exists(df_20.filepath.iloc[0])

# Xeno-Canto Extend by @vopani
df_xam = pd.read_csv(f'{CFG.BASE_PATH4}/train_extended.csv')
df_xam['filepath'] = CFG.BASE_PATH4 + '/A-M/' + df_xam.ebird_code + '/' + df_xam.filename
df_xnz = pd.read_csv(f'{CFG.BASE_PATH5}/train_extended.csv')
df_xnz['filepath'] = CFG.BASE_PATH5 + '/N-Z/' + df_xnz.ebird_code + '/' + df_xnz.filename
df_xc = pd.concat([df_xam, df_xnz], axis=0, ignore_index=True)
df_xc['primary_label'] = df_xc['ebird_code']
df_xc['scientific_name'] = df_xc['sci_name']
df_xc['common_name'] = df_xc['species']
df_xc['target'] = df_xc.primary_label.map(CFG.name2label2)
df_xc['birdclef'] = 'xc'
#assert tf.io.gfile.exists(df_xc.filepath.iloc[0])

# BirdCLEF-2021
df_21 = pd.read_csv(f'{CFG.BASE_PATH1}/train_metadata.csv')
df_21['filepath'] = CFG.BASE_PATH1 + '/train_short_audio/' + df_21.primary_label + '/' + df_21.filename
df_21['target'] = df_21.primary_label.map(CFG.name2label2)
df_21['birdclef'] = '21'
corrupt_paths = [f'{CFG.BASE_PATH1}/train_short_audio/houwre/XC590621.ogg',
                 f'{CFG.BASE_PATH1}/train_short_audio/cogdov/XC579430.ogg',
                 ]
df_21 = df_21[~df_21.filepath.isin(corrupt_paths)] # remove all zero audios
#assert tf.io.gfile.exists(df_21.filepath.iloc[0])

# BirdCLEF-2022
df_22 = pd.read_csv(f'{CFG.BASE_PATH2}/train_metadata.csv')
df_22['filepath'] = CFG.BASE_PATH2 + '/train_audio/' + df_22.filename
df_22['target'] = df_22.primary_label.map(CFG.name2label2)
df_22['birdclef'] = '22'
#assert tf.io.gfile.exists(df_22.filepath.iloc[0])

# Merge 2021 and 2022 for pretraining
df_pre = pd.concat([df_20, df_21, df_22, df_xc], axis=0, ignore_index=True)
df_pre['filename'] = df_pre.filepath.map(lambda x: x.split('/')[-1])
df_pre['xc_id'] = df_pre.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
nodup_idx = df_pre[['xc_id','primary_label','author']].drop_duplicates().index
df_pre = df_pre.loc[nodup_idx].reset_index(drop=True)

# # Remove duplicates
df_pre = df_pre[~df_pre.xc_id.isin(df_23.xc_id)].reset_index(drop=True)
corrupt_mp3s = json.load(open('/kaggle/input/birdclef-2023-dataset/corrupt_mp3_files.json','r'))
corrupt_mp3s = [path.replace('/kaggle/input','/kaggle/input/birdclef-2023-dataset') for path in corrupt_mp3s]
df_pre = df_pre[~df_pre.filepath.isin(corrupt_mp3s)]
df_pre = df_pre[['filename','filepath','primary_label','secondary_labels',
                 'rating','author','file_type','xc_id','scientific_name',
                'common_name','target','birdclef','bird_seen']]
# Display rows
print("# Samples for Pre-Training: {:,}".format(len(df_pre)))
df_pre.head(2).style.set_caption("Pre-Training Data").set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'blue'),
        ('font-size', '16px')
    ]
}])


df_pre['filepath']=df_pre['filepath'].str.replace('train_audio', 'train_audio_wav')
df_pre['filepath']=df_pre['filepath'].str.replace('train_short_audio', 'train_short_audio_wav')
df_pre['filepath']=df_pre['filepath'].str.replace('A-M', 'A-M_wav')
df_pre['filepath']=df_pre['filepath'].str.replace('mp3', 'wav')
df_pre['filepath']=df_pre['filepath'].str.replace('ogg', 'wav')
df_pre=df_pre.fillna('')

df_pre['out_filepath'] = df_pre.filepath
#df_pre['out_filepath'] = df_pre['out_filepath'].str.replace('ogg','wav')
#df_pre['out_filepath'] = df_pre['out_filepath'].str.replace('mp3','wav')
audio_dir_names = ['train_audio','train_short_audio','A-M','N-Z']
for audio_dir_name in audio_dir_names:
    df_pre['out_filepath'] = df_pre['out_filepath'].str.replace(audio_dir_name, f'{audio_dir_name}_mono')
df_pre['out_dir'] = df_pre['out_filepath'].str.extract(r'(.*\/)[^\/]+$', expand=False)
for out_dir in df_pre.out_dir.unique().tolist():
   os.makedirs(out_dir, exist_ok=True)
   
# Import required packages
from sklearn.model_selection import StratifiedKFold

# Initialize the StratifiedKFold object with 5 splits and shuffle the data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)

# Reset the index of the dataframe
df_pre = df_pre.reset_index(drop=True)

# Create a new column in the dataframe to store the fold number for each row
df_pre["fold"] = -1

# Iterate over the folds and assign the corresponding fold number to each row in the dataframe
for fold, (train_idx, val_idx) in enumerate(skf.split(df_pre, df_pre['primary_label'])):
    df_pre.loc[val_idx, 'fold'] = fold
    
train_dataset = prep.BirdDataset(df_pre, is_train=True)

def inference_collate(batch):
    features, sample_info = list(zip(*batch))
    features = torch.cat(features, dim=0)
    # info
    batched_info = {}
    
    # 最初のサンプルからすべてのデータ項目のキーを取得
    keys = [k for d in sample_info[0] for k in d.keys()]

    # キーに基づいてデータをバッチ化
    for key in keys:
        # 各キーに対応する値のリストを作成
        values = [d[key] for sample in sample_info for d in sample if key in d]

        # 値がテンソルであることを確認し、スタックしてバッチ化
        if isinstance(values[0], torch.Tensor):
            batched_info[key] = torch.stack(values, dim=0)
        # テンソルでない場合は、リストとして保存
        else:
            batched_info[key] = values
    return features, batched_info

def collate_fn(batch):
    features, sample_info = list(zip(*batch))
    
    return features, sample_info

ogg2wav_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers = os.cpu_count(),
    collate_fn=collate_fn
    )

from tqdm import tqdm

for idx, (features, sample_info) in enumerate(tqdm(ogg2wav_loader)):
    sys.stdout.flush()
    continue
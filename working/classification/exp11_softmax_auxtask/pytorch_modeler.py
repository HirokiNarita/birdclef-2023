############################################################################
# load library
############################################################################

# python default library
import os
import random

# general analysis tool-kit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data.dataset import Subset

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
from collections import defaultdict
import wandb

# original library
import common as com
import pytorch_preprocessing as prep
from pytorch_model import BirdCLEF23Net
from config import CFG
############################################################################
# Setting seed
############################################################################
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################
# Make Dataloader
############################################################################
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
    try :
        batched_info['target'] = torch.Tensor(batched_info['target']).long()
    except:
        pass
    return features, batched_info

def make_dataloder(train_dataset, valid_dataset):
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers = os.cpu_count(),
        )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers = os.cpu_count(),
        collate_fn = inference_collate,
        )
    
    return train_loader, valid_loader
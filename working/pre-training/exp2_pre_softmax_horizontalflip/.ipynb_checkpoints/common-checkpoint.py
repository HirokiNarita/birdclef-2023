import numpy as np
import pandas as pd
import sklearn.metrics

import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG

class AveragePrecisionScore:
    def __init__(self, multi_label=False):
        self.multi_label = multi_label

    def __call__(self, y_true, y_score):
        if self.multi_label:
            return sklearn.metrics.average_precision_score(y_true, y_score, average='samples')
        else:
            return sklearn.metrics.average_precision_score(y_true, y_score, average='macro')

def get_metrics():
    acc = lambda y_true, y_pred: (y_true == y_pred).sum().item() / y_true.size(0)
    auc = AveragePrecisionScore(multi_label=False)
    return acc, auc

def padded_cmap(y_true, y_pred, padding_factor=5):
    num_classes = y_true.shape[1]
    pad_rows = np.array([[1]*num_classes]*padding_factor)
    y_true = np.concatenate([y_true, pad_rows])
    y_pred = np.concatenate([y_pred, pad_rows])
    score = sklearn.metrics.average_precision_score(y_true, y_pred, average='macro',)
    return score

def get_criterion():
    if CFG.loss == "CCE":
        criterion = nn.CrossEntropyLoss()
    elif CFG.loss == "BCE":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(CFG.label_smoothing))
    else:
        raise ValueError("Loss not found")
    return criterion

def get_optimizer(model):
    if CFG.optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr=CFG.lr)
    else:
        raise ValueError("Optimizer not found")
    return opt

def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    
def upsample_data(df, thr=20):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df

def downsample_data(df, thr=500):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=CFG.seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df

def one_hot_encode(labels, num_classes=CFG.num_classes2):
    one_hot_encoded = np.zeros((labels.size, num_classes))
    one_hot_encoded[np.arange(labels.size), labels] = 1
    return one_hot_encoded
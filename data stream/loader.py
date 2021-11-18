import pandas as pd
import config
import torch
from sklearn.utils import shuffle
import numpy as np


def load_ann():
    source = pd.read_csv("./data/ann.csv")
    source = shuffle(source)
    width = source.shape[1]
    dataset_a1 = pd.DataFrame(columns=source.columns)
    dataset_a2 = pd.DataFrame(columns=source.columns)
    dataset_n = pd.DataFrame(columns=source.columns)
    for i in range(len(source)):
        label = source.iloc[i, width-1]
        if label == 1:
            dataset_a1 = dataset_a1.append(source.iloc[i, :])
        elif label == 2:
            dataset_a2 = dataset_a2.append(source.iloc[i, :])
        else:
            dataset_n = dataset_n.append(source.iloc[i, :])
    dataset_a1 = dataset_a1.reset_index(drop=True)
    dataset_a2 = dataset_a2.reset_index(drop=True)
    dataset_n = dataset_n.reset_index(drop=True)
    dataset_a1 = torch.tensor(dataset_a1.values)[:, :-1].float().to(config.device)
    dataset_a2 = torch.tensor(dataset_a2.values)[:, :-1].float().to(config.device)
    dataset_n = torch.tensor(dataset_n.values)[:, :-1].float().to(config.device)

    return dataset_a1, dataset_a2, dataset_n

def load_har():
    source = pd.read_csv("./data/har.csv")
    source = shuffle(source)
    width = source.shape[1]
    dataset_a1 = pd.DataFrame(columns=source.columns)
    dataset_a2 = pd.DataFrame(columns=source.columns)
    dataset_n = pd.DataFrame(columns=source.columns)
    for i in range(len(source)):
        label = source.iloc[i, width-1]
        if label == 1:
            dataset_a1 = dataset_a1.append(source.iloc[i, :])
        elif label == 2:
            dataset_a2 = dataset_a2.append(source.iloc[i, :])
        else:
            dataset_n = dataset_n.append(source.iloc[i, :])
    dataset_a1 = dataset_a1.reset_index(drop=True)
    dataset_a2 = dataset_a2.reset_index(drop=True)
    dataset_n = dataset_n.reset_index(drop=True)
    dataset_a1 = torch.tensor(dataset_a1.values)[:, :-1].float().to(config.device)
    dataset_a2 = torch.tensor(dataset_a2.values)[:, :-1].float().to(config.device)
    dataset_n = torch.tensor(dataset_n.values)[:, :-1].float().to(config.device)

    return dataset_a1, dataset_a2, dataset_n
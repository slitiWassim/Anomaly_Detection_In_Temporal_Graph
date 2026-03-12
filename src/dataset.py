

import pandas as pd
import numpy as np
import torch

from torch_geometric.datasets import JODIEDataset
from typing import Optional
from torch_geometric.transforms import BaseTransform

from .temporal_data import TemporalData

import os
class TemporalSplit(BaseTransform):
    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        key: Optional[str] = "t",
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.key = key

    def forward(self, data):
        key = self.key
        t = data[key].sort().values
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio
        val_time, test_time = np.quantile(
            t.cpu().numpy(), [1. - val_ratio - test_ratio, 1. - test_ratio])
        data.train_mask = data[key] < val_time
        data.val_mask = torch.logical_and(data[key] >= val_time, data[key]
                                          < test_time)
        data.test_mask = data[key] >= test_time
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.val_ratio}, '
                f'{self.test_ratio})')


def process_amazon(root):
    path = f'{root}/Reviews.csv'
    num_vote = 10

    df = pd.read_csv(path, delimiter=',')
    selected_columns = ['ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
    df = df[selected_columns]  

    # label edge
    # label principle: #vote>=10 & helpratio>=0.75 normal(0) & helpratio<=0.25 abnormal(1), left unknown(2)
    df['label'] = 2 
    df['HelpfulnessRatio'] = df['HelpfulnessNumerator']/df['HelpfulnessDenominator']
    df['HelpfulnessRatio'] = df['HelpfulnessRatio'].fillna(0)
    df.loc[(df['HelpfulnessDenominator'] >= num_vote) & (df['HelpfulnessRatio'] >= 0.75), 'label'] = 0 
    df.loc[(df['HelpfulnessDenominator'] >= num_vote) & (df['HelpfulnessRatio'] <= 0.25), 'label'] = 1
    df['label'] = df['label'].astype('float')

    # normalization
    # u,i,ts,label
    df['idx'] = df.index + 1
    df = df[['UserId', 'ProductId', 'Time', 'label', 'idx']]
    uid, uid_idx = np.unique(df['UserId'],return_inverse=True) 
    pid, pid_idx = np.unique(df['ProductId'],return_inverse=True) 
    num_u, num_p, nnodes = len(uid), len(pid), len(uid) + len(pid)
    df['UserId'] = uid_idx + 1 # start from 1
    df['ProductId'] = pid_idx + num_u + 1
    # sorted by ts
    df.rename(columns={'UserId': 'u', 'ProductId': 'i', 'Time': 'ts'}, inplace=True)
    df = df.sort_values(by='ts', ascending=True)
    df = df.reset_index(drop=True)
    df['idx'] = df.index + 1

    # save
    if not os.path.exists(f'{root}/amazon'):
        os.makedirs(f'{root}/amazon')
    df.to_csv(f'{root}/amazon/ml_amazon.csv')
    nnode = len(uid) + len(pid)
    nedge = len(df)  
    node_feat = np.zeros((nnode + 1, 172)) 
    edge_feat = np.zeros((nedge + 1, 172)) 
    print(node_feat.shape)# (330317+1, 172)
    print(edge_feat.shape)# (568454+1, 172)
    np.save(f'{root}/amazon/ml_amazon_node.npy', node_feat)
    np.save(f'{root}/amazon/ml_amazon.npy', edge_feat)
    print(f'num_u:{num_u}, num_p:{num_p}, num_nodes:{nnodes}, num_edges:{len(df)}')

def load_dataset(dataset, val_ratio= 0.15, test_ratio= 0.15, zero_edge=False):
    if 'amazon' in dataset:
        path = f"data/{dataset}/ml_amazon.csv"
        if not os.path.exists(path):
            process_amazon('data')
        graph_df = pd.read_csv(f"data/{dataset}/ml_amazon.csv")
        row = torch.from_numpy(graph_df.u.values).to(torch.long)
        col = torch.from_numpy(graph_df.i.values).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index - 1 
        labels = torch.from_numpy(graph_df.label.values).to(torch.float)
        stamps = graph_df.ts.values
        num_nodes = edge_index.max().item() + 1
        t = torch.from_numpy(stamps).long()
        data = TemporalData(
            src=edge_index[0], dst=edge_index[1], t=t, y=labels, num_nodes=num_nodes
        )
        data = TemporalSplit(val_ratio=val_ratio, test_ratio=test_ratio)(data) 
        data.x = torch.zeros(data.num_nodes, 1)  
        data.msg = torch.zeros(data.num_events, 1)
    
    elif dataset in ["otc", "alpha"]:
        graph_df = pd.read_csv(f"data/bitcoin{dataset}.csv")
        row = torch.from_numpy(graph_df.u.values).to(torch.long)
        col = torch.from_numpy(graph_df.i.values).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index - 1
        labels = torch.from_numpy(graph_df.label.values).to(torch.float)
        stamps = graph_df.ts.values
        num_nodes = edge_index.max().item() + 1
        t = torch.from_numpy(stamps).long()
        data = TemporalData(
            src=edge_index[0], dst=edge_index[1], t=t, y=labels, num_nodes=num_nodes
        )
        data = TemporalSplit(val_ratio=val_ratio, test_ratio=test_ratio)(data)
        data.x = torch.zeros(data.num_nodes, 1)  # assign zero node features
        data.msg = torch.zeros(data.num_events, 1)  # assign zero edge features

    else:
        data = JODIEDataset(root="data/", 
                            name=dataset, 
                            transform=TemporalSplit(val_ratio=val_ratio, test_ratio=test_ratio))[0]
        data = TemporalData(**data.to_dict())
        data.x = torch.zeros(data.num_nodes, 1)  
        if zero_edge:
            data.msg = torch.zeros(data.num_events, 1)  

    print("=" * 20, "Data statistics", "=" * 20)
    print(f"Name: {dataset}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_events}")
    print(f"Number of node features: {data.x.size(1)}")
    print(f"Number of edge features: {data.msg.size(1)}")
    num_lbl = (data.y!=2).sum().item()
    num_abn = (data.y==1).sum().item()
    print(f"Number of anomalies: {num_abn/num_lbl:.3%}")
    return data

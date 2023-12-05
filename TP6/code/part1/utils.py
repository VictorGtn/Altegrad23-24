"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()
    p1=.2
    p2=.4
    ############## Task 1
    
    ##################
    # your code here #
    ##################
    for i in range(50):
        n=randint(10,20)
        G = nx.erdos_renyi_graph(n,p1)
        Gs.append(G)
        y.append(0)

        n=randint(10,20)
        G = nx.erdos_renyi_graph(n,p2)
        Gs.append(G)
        y.append(1)

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

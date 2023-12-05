"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4
        
##################
# your code here #
##################
Gs = [nx.cycle_graph(i) for i in range(10, 20)]



############## Task 5
        
##################
# your code here #
##################
adj=sp.block_diag([nx.adjacency_matrix(G) for G in Gs])
x=np.ones((adj.shape[0],1))
idx=[]
for i,G in enumerate(Gs):
    idx.extend([i]*G.number_of_nodes())
idx=torch.LongTensor(idx).to(device)
adj=sparse_mx_to_torch_sparse_tensor(adj).to(device)
x=torch.FloatTensor(x).to(device)




############## Task 8
        
##################
# your code here #
##################
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
print('Here is the task 8 with aggr = sum and readout = sum:',model(x, adj, idx))
print('*'*50)
model = GNN(1, hidden_dim, output_dim, 'sum', 'mean', dropout).to(device)
print('Here is the task 8 with aggr = sum and readout = mean:',model(x, adj, idx))
print('*'*50)
model = GNN(1, hidden_dim, output_dim, 'mean', 'mean', dropout).to(device)
print('Here is the task 8 with aggr = mean and readout = mean:',model(x, adj, idx))

print('*'*50)
model = GNN(1, hidden_dim, output_dim, 'mean', 'sum', dropout).to(device)
print('Here is the task 8 with aggr = mean and readout = sum:',model(x, adj, idx))
print('*'*50)


############## Task 9
        
##################
# your code here #
##################
G1=nx.Graph()
G1.add_nodes_from([0,1,2,3,4,5])
G1.add_edges_from([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)])
G2=nx.Graph()
G2.add_nodes_from([0,1,2,3,4,5])
G2.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)])


############## Task 10
        
##################
# your code here #
##################
adj1 = nx.adjacency_matrix(G1)
adj2 = nx.adjacency_matrix(G2)
adj = sp.block_diag([adj1, adj2])
idx=[]
idx.extend([0]*G1.number_of_nodes())
idx.extend([1]*G2.number_of_nodes())
x=np.ones((adj.shape[0],1))
idx=torch.LongTensor(idx).to(device)
adj=sparse_mx_to_torch_sparse_tensor(adj).to(device)
x=torch.FloatTensor(x).to(device)

############## Task 11
        
##################
# your code here #
##################
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
print('Here is the task 11 :',model(x, adj, idx))



##Other grpah that are not distinguished by the GNN.

############## Task 9
        
##################
# your code here #
##################
G1=nx.Graph()
G1.add_nodes_from([0,1,2,3,4,5])
G1.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,5),(2,4)])
G2=nx.Graph()
G2.add_nodes_from([0,1,2,3,4,5])
G2.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5)])


############## Task 10
        
##################
# your code here #
##################
adj1 = nx.adjacency_matrix(G1)
adj2 = nx.adjacency_matrix(G2)
adj = sp.block_diag([adj1, adj2])
idx=[]
idx.extend([0]*G1.number_of_nodes())
idx.extend([1]*G2.number_of_nodes())
x=np.ones((adj.shape[0],1))
idx=torch.LongTensor(idx).to(device)
adj=sparse_mx_to_torch_sparse_tensor(adj).to(device)
x=torch.FloatTensor(x).to(device)

############## Task 11
        
##################
# your code here #
##################
model = GNN(1, hidden_dim, output_dim, 'sum', 'sum', dropout).to(device)
print('Two graph that cannot be distinguished :',model(x, adj, idx))
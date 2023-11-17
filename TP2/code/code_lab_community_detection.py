"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


G = nx.read_edgelist('datasets/CA-HepTh.txt', nodetype=int, comments='#', delimiter='\t')
largest_cc = max(nx.connected_components(G), key=len)
giant = G.subgraph(largest_cc).copy()

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    # your code here #
    ##################
    A = nx.adjacency_matrix(G)
    degrees = dict(G.degree())
    D = diags(list(degrees.values()), 0)
    I = eye(G.number_of_nodes())
    L = I - D.power(-1) @ A
    eigvals, eigvecs = eigs(L, k=k, which='SR')
    eigvecs = eigvecs.real
    eigvals = eigvals.real
    k_means = KMeans(n_clusters=k)
    k_means.fit(eigvecs)
    clustering = {n: c for n, c in zip(G.nodes(), k_means.labels_)}




    
    return clustering



############## Task 7

##################
# your code here #
##################

giant_clu= spectral_clustering(giant, 50)





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    ##################
    # your code here #
    ##################
    m = G.number_of_edges()
    nc=len(set(clustering.values())) #Gives us the number of clusters
    modularity=0 
    for i in range(nc):
        node_list = [n for n, v in clustering.items() if v == i]  # Get the nodes that belong to the i-th cluster
        community = G.subgraph(node_list)  # get subgraph that corresponds to current cluster
        lc = community.number_of_edges()
        degrees = dict(G.degree())
        dc = sum([degrees[node] for node in community.nodes()]) #get the degree of the nodes 
        modularity += lc/m - (dc/(2*m)) ** 2
    return modularity



############## Task 9

##################
# your code here #
##################

#first one
print('Modularity of the first clustering (Spectral Clutering): ', modularity(giant, giant_clu))

#second one : let's first do a random clustering 

def random_clustering(G, k):
    clustering = {}
    nodes = list(G.nodes())
    for node in nodes:
        clustering[node] = np.random.randint(0, k)
    return clustering

random_giant = random_clustering(giant, 50)
print('Modularity of the second clustering (Random Clustering) : ', modularity(giant, random_giant))








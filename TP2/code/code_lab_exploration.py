"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################
G = nx.read_edgelist('datasets/CA-HepTh.txt', nodetype=int, comments='#', delimiter='\t')
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
############## Task 2

##################
# your code here #
########m##########

con= nx.number_connected_components(G)

print("Number of connected components:", con)
largest_cc = max(nx.connected_components(G), key=len)
subgraph = G.subgraph(largest_cc).copy()
print("Number of nodes in the giant connected component:", subgraph.number_of_nodes())
print("Number of edges in the giant connected component:", subgraph.number_of_edges())

print('proportion of nodes in the giant connected component : ', 100*subgraph.number_of_nodes()/G.number_of_nodes(),'%')
print('proportion of edges in the giant connected component : ', 100*subgraph.number_of_edges()/G.number_of_edges(),'%')
############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
##################
# your code here #
##################
min_degree = np.min(degree_sequence)
max_degree = np.max(degree_sequence)
median_degree = np.median(degree_sequence)
mean_degree = np.mean(degree_sequence)

print("Minimum degree:", min_degree)
print("Maximum degree:", max_degree)
print("Median degree:", median_degree)
print("Mean degree:", mean_degree)




############## Task 4

##################
# your code here #
##################
plt.figure()
hist = nx.degree_histogram(G)
degrees = range(len(hist))
plt.bar(degrees,hist)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

plt.figure()
plt.loglog(degrees, hist, 'o')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

############## Task 5

##################
# your code here #
##################

cc = nx.transitivity(G)
print("Global clustering coefficient:", cc)
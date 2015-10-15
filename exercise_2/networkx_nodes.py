#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph() # create a empty graph instance
graph = []
#dataset = 'datasets/drosophila_PPI.txt'
dataset = 'datasets/dataset.txt' #load the files
with open(dataset, 'r') as f:
    for line in f.readlines():
        graph.append(tuple(int(x) for x in line.strip().split())) # add the nodes to a list contain [(0,1), (2,3) ,(4,5)....]
G.add_edges_from(graph) # Add this list to Graph

print nx.degree(G)
print nx.betweenness_centrality(G, normalized=False)
print [v for v in nx.clustering(G).values()]
print nx.transitivity(G)
#nx.draw_networkx(G) #this doesn't work well for large datasets, but it is feasible for small graph
#plt.show()

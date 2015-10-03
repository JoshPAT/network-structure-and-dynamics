#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
graph = []
#input the dataset:
#dataset = 'processed_dataset_inet.txt'
dataset = 'datasets/inet.txt'
with open(dataset, 'r') as f:
    for line in f.readlines():
        graph.append(tuple(int(x) for x in line.strip().split()))

G.add_edges_from(graph)
#print nx.betweenness_centrality(G, normalized=False)
#nx.draw_networkx(G)
#plt.show()
print [v for v in nx.clustering(G).values()]
print nx.transitivity(G)
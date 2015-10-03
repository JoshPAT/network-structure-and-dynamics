#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import networkx as nx
import matplotlib.pyplot as plt
import time


G = nx.Graph()
graph = []
#dataset = 'datasets/drosophila_PPI.txt'
dataset = 'datasets/dataset.txt'
with open(dataset, 'r') as f:
    for line in f.readlines():
        graph.append(tuple(int(x) for x in line.strip().split()))
start_time = time.time()
G.add_edges_from(graph)
print nx.betweenness_centrality(G, normalized=False)
print time.time() - start_time
#nx.draw_networkx(G)
#plt.show()
#print [v for v in nx.clustering(G).values()]
#print nx.transitivity(G)

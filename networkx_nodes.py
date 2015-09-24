#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
graph = []
#input the dataset:
dataset = raw_input('Type the name of your dataset: ')
with open(dataset, 'r') as f:
    for line in f.readlines():
        graph.append(tuple(line.strip().split()))

G.add_edges_from(graph)
nx.draw_networkx(G)
plt.show()



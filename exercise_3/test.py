#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import networkx as nx

G = nx.Graph() # create a empty graph instance
graph = []
#dataset = 'datasets/drosophila_PPI.txt'
dataset = 'datasets/erdos_renyi.txt' #load the files
with open(dataset, 'r') as f:
    for line in f.readlines():
        graph.append(tuple(int(x) for x in line.strip().split())) # add the nodes to a list contain [(0,1), (2,3) ,(4,5)....]
G.add_edges_from(graph) # Add this list to Graph

print nx.average_shortest_path_length(G)
#!/usr/bin/env python 2.7.8
# -*- coding: utf-8 -*-

'''
This program is about Part 4 'Breadth first search'.
'''
import math, operator, itertools
import matplotlib.pyplot as plt

try:
    import nsd
except ImportError:
    print 'make sure file nsd.py is in same folder'


def array_list(dataset):
    n = int(nsd.compute_node_number(dataset))
    d = {}
    for x in xrange(n+1):
        d[x] = []
    with open(dataset, 'r') as f:
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')]
            d[i].append(j)
            d[j].append(i)
    return d

# exercise 11
def BFS(d, s):
    discovered = {}
    level = [s]
    stage = 1
    discovered[s] = 0
    while len(level) > 0:
        next_level = []
        for u1 in level:
            for u2 in d[u1]:
                if u2 not in discovered:
                    discovered[u2] = stage
                    next_level.append(u2)
        level = next_level
        stage += 1
    sorted_discovered = sorted(discovered.items(), key = operator.itemgetter(1))
    return ['%d => %d ' % (k, v ) for v, k in sorted_discovered]

# exercise 12
@nsd.run_time
def compute_size(dataset):
    d = array_list(dataset)
    discovered = [None] * len(d.keys()) 
    for x,_ in enumerate(discovered):
        if discovered[x] is None:
            if not d[x]:
                discovered[x] = 'connectless'
            else:
                level = [x]
                while len(level) > 0:
                    next_level = []
                    for u1 in level:
                        for u2 in d[u1]:
                            if discovered[u2] is None:
                                discovered[u2] = x
                                next_level.append(u2)
                    level = next_level
    component = {}
    for k, g in itertools.groupby(sorted(discovered)):
        if k != 'connectless':
            size = len(list(g))
            component[k] = size
            print 'component: %s, size: %s' % (k, size)
    sorted_componet = sorted(component.items(), key = operator.itemgetter(0))
    root = sorted_componet[0][0]
    size = sorted_componet[0][1]
    
    print 'biggest component: %d, size: %d' % (root, size)
    biggest_component = []
    for x,_ in enumerate(discovered):
        if discovered[x] == root:
            biggest_component.append(x)
    print biggest_component
    return component

# exercise 14
@nsd.run_time
def set_of_shortest_paths(dataset):
    d = array_list(dataset)
    discovered = {x:[] for x in d.keys()}
    stage_table = [0] * len(d.keys())
    for x,_ in enumerate(discovered):
        if len(discovered[x]) == 0:
            if not d[x]:
                discovered[x] = 'disconnected'
                stage_table[x] = 'disconnected'
            else:
                level = [x]
                discovered[x].append(0)
                while len(level) > 0:
                    next_level = []
                    for u1 in level:
                        for u2 in d[u1]:
                            if len(discovered[u2]) == 0:
                                stage_table[u2] = stage_table[u1] + 1
                                discovered[u2].append(u1)
                                next_level.append(u2)
                            elif len(discovered[u2]) > 0:
                                if stage_table[u2] == stage_table[u1] + 1:
                                    if u1 not in discovered[u2]:
                                        discovered[u2].append(u1)
                    level = next_level
        print discovered
        print stage_table
        return [discovered, stage_table]

# exercise 15
@nsd.run_time
def number_of_shortest_path(dataset , v):
    d = array_list(dataset)
    discovered = {x:[] for x in d.keys()}
    stage_table = [0] * len(d.keys())
    if len(discovered[v]) == 0:
            if not d[v]:
                discovered[v] = 'disconnected'
                stage_table[v] = 'disconnected'
            else:
                level = [v]
                discovered[v].append(-1)
                while len(level) > 0:
                    next_level = []
                    for u1 in level:
                        for u2 in d[u1]:
                            if len(discovered[u2]) == 0:
                                stage_table[u2] = stage_table[u1] + 1
                                discovered[u2].append(u1)
                                next_level.append(u2)
                            elif len(discovered[u2]) > 0:
                                if stage_table[u2] == stage_table[u1] + 1:
                                    if u1 not in discovered[u2]:
                                        discovered[u2].append(u1)
                    level = next_level
    print discovered
    print stage_table


@nsd.run_time
def compute_bfs(dataset, s):
    dt = array_list(dataset)
    return BFS(dt, s)

if __name__ == "__main__":
    print compute_bfs('processed_dataset.txt', 0)
    c1 = compute_size('processed_dataset_inet.txt')
    c2 = compute_size('processed_dataset_sophia.txt')
    set_of_shortest_paths('processed_dataset.txt')
    number_of_shortest_path('processed_dataset.txt', 3)

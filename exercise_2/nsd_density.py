#!/usr/bin/env python 2.7.8
# -*- coding: utf-8 -*-

'''
This program is about Part 3 'Local density vs global density'.
'''
import math
import matplotlib.pyplot as plt

try:
    import nsd
except ImportError:
    print 'make sure file nsd.py is in same folder'

# used to count combination (n r)
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

# exercise 10
@nsd.run_time
def compute_cluster_coefficient(dataset):
    n = int(nsd.compute_node_number(dataset))
    d = {}
    for x in xrange(n+1):
        d[x] = []
    cc = []
    with open(dataset, 'r') as f:
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')]
            d[i].append(j)
            d[j].append(i)
    nv = 0
    for k, v in d.iteritems():
        if v == []:
            cc.append('Undefined') # cc of node without connection should be undefined
        elif len(v) == 1:
            cc.append(0.0)
        else:
            for u1 in v:
                for u2 in d[u1]:
                    if u2 in v:
                        nv += 1
            nv = nv / 2.0 # (u1, u2) and (u2, u1) should be same, my algorithm counts twice,so here merge them
            cc.append((2 * nv)/ (len(v) * (len(v) - 1) )) # 
            nv = 0
    return cc

# exercise 10
@nsd.run_time
def compute_triangle_number(dataset):
    n = int(nsd.compute_node_number(dataset))
    d = {}
    for x in xrange(n+1):
        d[x] = []
    with open(dataset, 'r') as f:
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')]
            d[i].append(j)
            d[j].append(i)
    nv = 0
    ntr = 0
    for k, v in d.iteritems():
        if v == []:
            ntr += 0
        elif len(v) == 1:
            ntr += 0
        else:
            ntr += nCr(len(v), 2)
            for u1 in v:
                for u2 in d[u1]:
                    if u2 in v:
                        nv += 1
    nv= nv / 2.0 # (u1, u2, u3) and (u2, u1, u3) and (u3, u2, u1) should be same, my algorithm counts triple,so here merge them
    return nv / ntr

if __name__ == "__main__":
    print compute_cluster_coefficient('processed_dataset_inet.txt')
    print compute_triangle_number('processed_dataset_inet.txt')
    print compute_cluster_coefficient('processed_dataset_sophia.txt')
    print compute_triangle_number('processed_dataset_sophia.txt')
    

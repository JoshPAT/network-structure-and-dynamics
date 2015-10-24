#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import numpy as np
import os, functools
import itertools
from linkprediction import run_time
import collections
import networkx as nx
import matplotlib.pyplot as plt


pair_sets = collections.defaultdict(list)


print pair_sets.keys()
print [14,21] in pair_sets.keys()

with open('outputs/missed_links.txt', 'r') as f:
    a = np.array(
        [line.strip().split(' ') for line in f.readlines()],
        dtype = np.int16
        )
    a.sort()
with open('outputs/karz_method/results.txt', 'r') as f:
    for line in f.readlines():
        i, j, s = line.strip().split(' ')
        i, j = int(i), int(j)
        if i > j: i, j = j, i
        if any(np.equal(a,[i,j]).all(1)):
            print i, j, s


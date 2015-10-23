#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import numpy as np
import os, functools
import itertools
from linkprediction import run_time
import collections
'''
a = np.array([[1, 3],[2,3],[4,5]])

print a[-1:]

b = np.array([1, 3])

filepath = os.path.split(os.path.abspath(__file__))[0]
dirpath = os.path.join(filepath, 'datasets/')

print set(itertools.combinations(np.array([1,2,3,4]) ,2))         


for x in list(itertools.combinations(np.array([1,2,3,4]) ,2)):
    print type(set(x))
'''
print tuple([2,3])
nodes_number =1000

@run_time
def test_1():
    s =0
    for i in xrange(nodes_number+1):
        for j in xrange(i+1, nodes_number+1):
            s += sum([i,j])

@run_time
def test_2():
    s =0
    for i, j in itertools.combinations(xrange(nodes_number + 1),2):
        s += sum([i,j])
  
test_1()
test_2()  
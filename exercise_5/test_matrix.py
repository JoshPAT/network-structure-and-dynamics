#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import itertools
import operator
import collections
import numpy as np
from scipy.sparse import *
    def tbf_strategy(self, start_i, trials = 0):
        '''
        Predicts the links by the maximal sum of two nodes.
        '''
        #print self.node_in_order
        #print self.degree_in_order
        _path = os.path.join(self.outputs_path, 'tbf_strategy')
        with open(_path, 'w') as f:
            for k, v in sorted(self.random_init_edges.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))
        i = start_i # default value 1000
        index_1, index_2 = 0, 1 # start from 1,2
        # move the the most right
        max_len = self.node_number - 1
        while 1:
            if self.degree_in_order[max_len] > 0:
                break
            del self.degree_in_order[max_len]
            del self.node_in_order[max_len]
            max_len -= 1
        print max_len        
        dict_links = collections.defaultdict(set)
        matrix_links = np.zeros((max_len, max_len))
        #matrix_links = collections.defaultdict(int)
        for index_1, index_2 in itertools.combinations(xrange(max_len),2):
            matrix_links[(index_1, index_2)] = self.degree_in_order[index_1] + self.degree_in_order[index_2] 
            dict_links[index_1].add(index_2)
            dict_links[index_2].add(index_1)
        with open(_path, 'a') as f:
            while 1:
                try:
                    if i > trials:
                        break
                    index_1, index_2 = np.argwhere(matrix_links == matrix_links.max())[0]
                    #index_1, index_2 = max(matrix_links.iteritems(), key = operator.itemgetter(1))[0]
                    u = self.node_in_order[index_1]
                    v = self.node_in_order[index_2]
                    dict_links[index_2].remove(index_1)
                    dict_links[index_1].remove(index_2)
                    if u not in self.links_tested[v]:
                        if self._measure_primitive(u, v):
                            f.write('%d %d %d\n' % (i, u, v))
                            for index_n in dict_links[index_1]:
                                if index_n < index_1:
                                    matrix_links[(index_n,index_1)] += 1
                                else:
                                    matrix_links[(index_1,index_n)] += 1
                            for index_n in dict_links[index_2]:
                                if index_n < index_2:
                                    matrix_links[(index_n,index_2)] += 1
                                else:
                                    matrix_links[(index_2,index_n)] += 1
                    print index_1, index_2, matrix_links[(index_1, index_2)]
                    #del matrix_links[index_1, index_2]
                    matrix_links[index_1, index_2] = 0
                    i += 1
                except KeyError:
                    break

    '''
    def tbf_strategy(self, start_i, trials = 0):
        
        Predicts the links by the maximal sum of two nodes.
        
        #print self.node_in_order
        #print self.degree_in_order
        _path = os.path.join(self.outputs_path, 'tbf_strategy')
        with open(_path, 'w') as f:
            for k, v in sorted(self.random_init_edges.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))
        i = start_i # default value 1000
        tested_links = self.sample_nodes_edges.copy()
        save_links = collections.defaultdict(int)
        max_len = self.node_number
        while 1:
            if self.degree_in_order[max_len] > 0:
                break
            del self.degree_in_order[max_len]
            del self.node_in_order[max_len]
            max_len -= 1
        print max_len 
        node_tested = dict.fromkeys(self.node_in_order.values(), max_len)
        index_1, index_2 = 0, 1 # start from 1,2
        u = self.node_in_order[index_1]
        v = self.node_in_order[index_2]
        with open(_path, 'a') as f:
            while 1:
                try:
                    if i > trials:
                        break
                    if index_1 < self.node_number:
                        if u not in self.links_tested[v]:
                            if self._measure_primitive(u, v):
                                f.write('%d %d %d\n' % (i, u, v))
                                #index_1, index_2 = self._alter_twonodes(index_1, index_2)
                            self.links_tested[u].append(v)
                            self.links_tested[v].append(u)
                            i += 1
                        while index_2 > max_len - 1:
                            #print index_1, index_2
                            u, v = max(save_links.iteritems(), key = operator.itemgetter(1))[0]
                            index_1 = self.node_in_order.keys()[self.node_in_order.values().index(u)]
                            index_2 = self.node_in_order.keys()[self.node_in_order.values().index(v)]
                            del save_links[(u, v)]
                        save_links, index_1, index_2 = self._save_links(save_links, index_1, index_2)
                        u, v = max(save_links.iteritems(), key = operator.itemgetter(1))[0]
                        #print save_links
                        index_1 = self.node_in_order.keys()[self.node_in_order.values().index(u)]
                        index_2 = self.node_in_order.keys()[self.node_in_order.values().index(v)]
                        #print u, v
                        #print index_1, index_2
                        del save_links[(u, v)]
                except ValueError:
                    break
    '''
d = {1:2, 3:4, 5:6}
index = d.values()[]




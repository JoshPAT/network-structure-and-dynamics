#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import os
import time
import functools
import argparse
import logging
import itertools
import collections
import math
import random
import operator
import numpy as np

def run_time(func):
    '''
    A dectorator used to compute run time in a fucntion
    '''
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print "Computation Time of %s: %s" % \
              (func.__name__.capitalize(), end_time - start_time)
        return result
    return wrapper

DATASETS = {'o': 'Flickr', 't': 'Flickr-test', 'i' : 'inet' }

def nCr(n,r):
    '''
    Return math combination nCr
    '''
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        nargs='?',
        type=str,
        help="Select the dataset your want to create the file. \
                Default is 'Flickr-test'",
        default='t'
    )
    parser.add_argument(
        "-r",
        "--random",
        nargs='?',
        type=int,
        help="",
        default=2000
    )
    parser.add_argument(
        "-t",
        "--tests",
        nargs='?',
        type=int,
        help="",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eff",
        action="store_true",
        help="",
    )

    return parser.parse_args()

class Graph(object):
    '''
    a graph contains infos about node number, degree, neighours tables of node,
    degree distrubition, and its cumlative degree distribution.
    '''
    def __init__(self, dataname = None):
        self.outputs_path = os.path.join('outputs/', dataname, 'graphs/')
        self.processed_dataset = self._process_dataset(dataname)
        self.node_number = self._compute_node_number()
        self.degree_table = self._compute_node_degree()
        self.nodes_edges = self._compute_nodes_edges()
        self.degree_distribution = self._compute_degree_distribution()

    def _process_dataset(self, dataname):
        '''
        Return the datasets by deleting the possible loops in the original links.
        '''
        datasets_path = os.path.join('datasets/', dataname)
        if not os.path.exists(self.outputs_path):
            os.makedirs(self.outputs_path)
        processed_dataname = os.path.join(self.outputs_path, 
                                          os.path.basename(datasets_path))
        dataset_list = []
        with open(datasets_path, 'r') as f1:
            with open(processed_dataname, 'w') as f2:
                for line in f1.readlines():
                    i, j = [int(x) for x in line.strip().split(' ')]
                    if i != j:
                        dataset_list.append((i,j))
                        f2.write('%s %s\n' % (i, j))
        return dataset_list

    def _compute_node_number(self):
        '''
        Return the node number.
        '''
        n = max([max(i,j) for i, j in self.processed_dataset])
        with open(os.path.join(self.outputs_path, 'graph.n'), 'w') as f:
            f.write(str(n))
        return n

    def _compute_node_degree(self):
        '''
        Return node degree table.
        '''
        degree_table = [0] * (self.node_number + 1)
        for i, j in self.processed_dataset:
            degree_table[i] += 1
            degree_table[j] += 1
        with open(os.path.join(self.outputs_path, 'graph.deg'), 'w') as f:
            for n in degree_table:
                f.write('%s\n' % n)
        return degree_table

    def _compute_nodes_edges(self):
        '''
        Return nodes edges in lists.
        '''
        nodes_edges = collections.defaultdict(list)
        for i, j in self.processed_dataset:
            nodes_edges[i].append(j)
            nodes_edges[j].append(i)  
        return nodes_edges

    def _compute_degree_distribution(self):
        '''
        Return the degree distribution.
        '''
        degree_distribution = collections.defaultdict(int)
        for degree in self.degree_table:
            degree_distribution[degree] += 1
        with open(os.path.join(self.outputs_path, 'graph.dn'), 'w') as f:
            for degree, node in degree_distribution.iteritems():
                f.write('%s %s\n' % (degree , node))
        return degree_distribution
    
    def _cumlative_degree_distribution(self):
        '''
        Return the cumlative degree distribution.
        '''
        cum_degree_distribution = self.degree_distribution.copy()
        n = 0 #inital nodes = 0
        for degree in reversed(sorted(cum_degree_distribution.keys())):
            n = cum_degree_distribution[degree] + n # add nodes has bigger degree to previous nodes
            cum_degree_distribution[degree] = n
        return cum_degree_distribution

    def _compute_triangle_values(self):
        '''
        Return Global Transitive Ratio & Global Clustering Coefficient.
        '''
        value_cc = {}
        num_v, num_tri, sum_tri = 0, 0, 0
        for node in self.nodes_edges.keys():
            if len(self.nodes_edges[node]) <= 1:
                value_cc[node] = 0 
            else:
                num_v += nCr(len(self.nodes_edges[node]), 2)
                for neighbour in self.nodes_edges[node]:
                    for neighbour_of_neighbour in self.nodes_edges[neighbour]:
                        if neighbour_of_neighbour in self.nodes_edges[node]:
                            num_tri += 1
                # (u1, u2) and (u2, u1) should be same, my algorithm counts twice,so here merge them
                num_tri = num_tri / 2.0
                sum_tri += num_tri
                value_cc[node] = (2 * num_tri)/ (len(self.nodes_edges[node]) * (len(self.nodes_edges[node]) - 1) )
                num_tri = 0
        average_tr = sum_tri / num_v
        average_cc = (sum(value_cc.values()) / self.node_number)
        return average_tr, average_cc

    def _average_clustering(self, trials = 1000):
        '''
        Return the approximations of Average clustering.
        '''
        triangles = 0
        for node in [random.randint(0, self.node_number) for _ in xrange(trials)]:
            neighbours = self.nodes_edges[node]
            if len(neighbours) < 2:
                continue
            u, v = random.sample(neighbours, 2)
            if u in self.nodes_edges[v]:
                triangles += 1
        return triangles/ float(trials)

    def graph_infos(self):
        '''
        Print all the characteriscs of the graph.
        '''
        print 'Numbers of degree 0: %s' % self.degree_table.count(0)
        print 'Max Degree: %s' % max(self.degree_table)
        print 'Min Degree: %s' % min(self.degree_table)
        print 'Average Degree: %s' % (sum(self.degree_table) * 1.0 / len(self.degree_table))
        print 'Density: %0.11f' % (2.0 * sum(self.degree_table) / (self.node_number * (self.node_number - 1)))
        print 'Approximations of average clustering coefficient: %0.11f' % self._average_clustering()
        #print 'transitive ratio: %0.11f\naverage clustering coefficient: %0.11f' % (self._compute_triangle_values())

class Simulation(Graph):

    def __init__(self, dataname, random_trials):
        super(Simulation, self).__init__(dataname)
        self.dataname = dataname
        self.sample_nodes_edges = self._clear_inputs(dataname)
        self.links_tested = collections.defaultdict(list)
        self._number_of_links()
        self.random_init_edges, self.start_i = self.random_phase(trials=random_trials)
        self.node_in_order = dict(enumerate(self._degree_in_order()[0]))
        self.degree_in_order = dict(enumerate(self._degree_in_order()[1]))

    def _clear_inputs(self, dataname):
        '''
        Return the bare structure of the original graphs and Make a new dir to save.
        '''
        self.outputs_path = os.path.join('outputs/', dataname, 'strategies/')
        if not os.path.exists(self.outputs_path):
            os.makedirs(self.outputs_path)
        sample_nodes_edges = {k: [] for k in self.nodes_edges}
        return sample_nodes_edges

    def _number_of_links(self):
        nb_links = os.path.join(self.outputs_path, 'links')
        with open(nb_links, 'w') as f:
            f.write('%s' % len(self.processed_dataset))

    def _measure_primitive(self, u, v):
        '''
        Return True if there is a link between node u and node v.
        Update the links in bare structure.
        '''
        if v in self.nodes_edges[u] and v not in self.sample_nodes_edges[u]:
            self.sample_nodes_edges[u].append(v)
            self.sample_nodes_edges[v].append(u)
            return True
        return False

    def _pure_random_process(self, i, links_found):
        u, v = random.sample(self.nodes_edges.keys(), 2)
        if u not in self.links_tested[v]:
            if self._measure_primitive(u, v):
                links_found[i] = (u, v)
            self.links_tested[u].append(v)
            self.links_tested[v].append(u)
            i += 1
        return i, links_found

    def _v_random_process(self, i, links_found):
        u, v = random.sample(self.nodes_edges.keys(), 2)
        if u not in self.links_tested[v]:
            if self._measure_primitive(u, v):
                links_found[i] = (u, v)
                for w in self.sample_nodes_edges[u]:
                    if v not in self.links_tested[w]:
                        if self._measure_primitive(v, w):
                            links_found[i] = (v, w)
                        self.links_tested[w].append(v)
                        self.links_tested[v].append(w)
                        i += 1
                for w in self.sample_nodes_edges[v]:
                    if u not in self.links_tested[w]:
                        if self._measure_primitive(u, w):
                            links_found[i] = (u, w)
                        self.links_tested[w].append(u)
                        self.links_tested[u].append(w)
                        i += 1
            self.links_tested[u].append(v)
            self.links_tested[v].append(u)
            i += 1
        return i, links_found

    def random_phase(self, trials = 1000):
        '''
        Return the links_found using random strategy.
        If "base" option is enabled, it means data generated is random base.
        Else it means data generated is for random_strategy.
        
        After that both will store the result in
        format {t, u, v} t stands for the times when links(u,v) is found.
        '''
        links_found = collections.defaultdict(list)
        links_need = len(self.processed_dataset) * 0.1 / 100
        while 1:
            i = 0
            while i < trials:
                i, links_found = self._v_random_process(i, links_found)
            links_found = {k: v for k,v in links_found.iteritems() if k < trials}
            if len(links_found.keys()) > links_need:
                break
            else:
                while len(links_found.keys()) < links_need:
                    i, links_found = self._v_random_process(i, links_found)
                break
        with open(os.path.join(self.outputs_path, 'random_base'), 'w') as f:
            for k, v in sorted(links_found.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))
        return links_found, i

    def random_strategy(self, start_i, trials = 0):
        i = start_i
        links_found = self.random_init_edges
        while i < trials:
            i, links_found = self._pure_random_process(i, links_found)
        with open(os.path.join(self.outputs_path, 'random_strategy'), 'w') as f:
            for k, v in sorted(links_found.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))

    def v_random_strategy(self, start_i, trials = 0):
        i = start_i
        links_found = self.random_init_edges
        while i < trials:
            i, links_found = self._v_random_process(i, links_found)
        with open(os.path.join(self.outputs_path, 'v_random_strategy'), 'w') as f:
            for k, v in sorted(links_found.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))

    def _degree_in_order(self):
        '''
        Return the lists nodes and lists of nodes' degree in desending order.
        '''
        degree_in_order = {k: len(v) for k,v in self.sample_nodes_edges.iteritems()}
        l = sorted(degree_in_order.iteritems(), key=operator.itemgetter(1), reverse=True)
        return zip(*l)

    def _max_degree(self):
        '''
        Return the node number have maximal degree.
        '''
        return max(self.sample_nodes_edges, 
                       key=lambda k: len(self.sample_nodes_edges[k]))

    def complete_strategy(self, start_i, trials = 0):
        '''
        Predicts the links by the maximal degree of the node.
        '''
        _path = os.path.join(self.outputs_path, 'complete_strategy')
        with open(_path, 'w') as f:
            for k, v in sorted(self.random_init_edges.iteritems()):
                f.write('%d %d %d\n' % (k, v[0], v[1]))
        i = start_i # default value 1000
        with open(_path, 'a') as f:
            while self.sample_nodes_edges.keys():
                _max = self._max_degree()
                for n in self.nodes_edges.keys():
                    if n not in self.links_tested[_max]:
                        if self._measure_primitive(_max, n):
                            f.write('%d %d %d\n' % (i, _max, n))
                        self.links_tested[_max].append(n)
                        self.links_tested[n].append(_max)
                        i += 1 
                del self.sample_nodes_edges[_max]
                if i > trials:
                    break

    def _alter_twonodes(self, index_1, index_2):
        '''
        Change the position of nodes according to their degree.
        Move the position of bigger degree to left.
        '''

        self.degree_in_order[index_1] += 1
        self.degree_in_order[index_2] += 1
        
        index_left = index_1 - 1
        while index_left > 0:
            if self.degree_in_order[index_1] > self.degree_in_order[index_left]:
                index_left -= 1
            else:
                break
        if index_left < index_1 - 1:
            self.degree_in_order[index_1], self.degree_in_order[index_left] \
                            = self.degree_in_order[index_left], self.degree_in_order[index_1] 
            self.node_in_order[index_1], self.node_in_order[index_left] \
                            = self.node_in_order[index_left], self.node_in_order[index_1]
            index_1, index_left = index_left, index_1
            logging.info("There is a switch between (%d, %d)" , index_1, index_left)
        
        index_left = index_2 -1
        while index_left > 1:
            if self.degree_in_order[index_2] > self.degree_in_order[index_left]:
                index_left -= 1
            else:
                break
        if index_left < index_2 -1:
            self.degree_in_order[index_2], self.degree_in_order[index_left] \
                            = self.degree_in_order[index_left], self.degree_in_order[index_2] 
            self.node_in_order[index_2], self.node_in_order[index_left] \
                            = self.node_in_order[index_left], self.node_in_order[index_2]
            #logging.info("There is a switch between (%d, %d)" , index_2, index_left)
        return index_1, index_2

    def _save_links(self, save_links, index_1, index_2):
        node_1, node_2 = self.node_in_order[index_1], self.node_in_order[index_2+1]
        save_links[(node_1, node_2)] = self.degree_in_order[index_1] + self.degree_in_order[index_2+1]
        if index_2 - index_1 > 1:
            node_3, node_4 = self.node_in_order[index_1+1], self.node_in_order[index_2]
            save_links[(node_3, node_4)] = self.degree_in_order[index_1+1] + self.degree_in_order[index_2]
            if self.degree_in_order[index_1+1] + self.degree_in_order[index_2] > \
                    self.degree_in_order[index_1] + self.degree_in_order[index_2+1]:
                return save_links, index_1+1, index_2
        return save_links, index_1, index_2+1

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

class Effiency(Graph):
    def __init__(self, dataname):
        super(Effiency, self).__init__(dataname)
        self.path = os.path.join('outputs/', dataname, 'strategies/')
        print self.path

    def analyse_reference_graph(self, test_number):
        existing_edges = sum(self.degree_table)
        possible_edges = (self.node_number * (self.node_number - 1)) / 2.0
        #print existing_edges
        #print possible_edges
        efficiency_worst = 0
        for i in xrange(test_number):
            efficiency_worst += max(min((i+1-possible_edges), existing_edges), 0)

        efficiency_best = 0
        for i in xrange(test_number):
            efficiency_best += min(i+1, existing_edges)

        efficiency_random = 0
        number_of_discovered_edges = 0
        for i in xrange(test_number):
            probability = (existing_edges-number_of_discovered_edges) * 1.0/(possible_edges-i)
            number_of_discovered_edges += probability
            efficiency_random += int(number_of_discovered_edges)

        return efficiency_worst, efficiency_best, efficiency_random
    
    def calculate_efficiency(self):
        existing_edges = sum(self.degree_table)
        possible_edges = (self.node_number * (self.node_number - 1)) / 2.0
        for outputs in ['random_strategy', 'v_random_strategy', 'complete_strategy', 'tbf_strategy',]:
            with open(os.path.join(self.path, outputs), 'r') as f:
                efficiency = 0
                i = 0
                index = 0
                for line in f.readlines():
                    next_index = int(line.strip().split(' ')[0])
                    efficiency += (next_index - index) * i
                    i += 1
            efficiency_worst, efficiency_best, efficiency_random = self.analyse_reference_graph(50000)
            efficiency_normalised = (efficiency - efficiency_worst) * 1.0 /(efficiency_best - efficiency_worst)
            efficiency_normalised_random = (efficiency_random - efficiency_worst) * 1.0 /(efficiency_best - efficiency_worst)
            efficiency_relative = efficiency_normalised * 1.0/ efficiency_normalised_random
            print outputs, int(efficiency_normalised), int(efficiency_relative)

def lazy_type(base_trials, tests):
    '''
    Just a simple fucntion to alter less and control more in commandline.
    '''
    c = Simulation(d, base_trials)
    c.graph_infos()
    c1, c2, c3 = copy.deepcopy(c), copy.deepcopy(c), copy.deepcopy(c)
    
    logging.info('Random strategy...')
    c.random_strategy(c.start_i,trials=tests)
    
    logging.info('V_random strategy...')
    c1.v_random_strategy(c.start_i,trials=tests)

    logging.info('Complete strategy...')
    c2.complete_strategy(c.start_i, trials=tests)
    
    logging.info('TBF strategy...')
    c3.tbf_strategy(c.start_i, trials=tests)

if __name__ == '__main__':
    import copy
    # Set the log level
    logging.basicConfig(level=logging.INFO)
    cmd = parse_args()
    # Find the data
    if not cmd.eff:
        d = DATASETS[cmd.dataset]
        lazy_type(cmd.random, cmd.tests)
    else:
        d1 = Effiency(DATASETS[cmd.dataset])
        print d1.analyse_reference_graph(50000)
        d1.calculate_efficiency()



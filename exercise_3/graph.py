#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import functools ,time, itertools, operator, math, datasets

# a dectorator used to compute run time in a fucntion
def run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print "Computation Time of %s: %s" % (func.__name__.capitalize(), end_time - start_time)
        return result
    return wrapper

# used to count combination (n r)
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class Graph(object):
    '''
    a graph contains infos about node number, degree, neighours tables of node,
    degree distrubition, and its cumlative degree distribution.
    '''
    def __init__(self, dataset = None):
        self.datasetpath = datasets.__path__[0] + '/'  + dataset.split('.')[0]
        self.dataset = []
        self.process_dataset(self.datasetpath + '.txt')

        self.node_file = self.datasetpath + '_graphe.n'
        self.node_number = 0
        self.compute_node_number()

        self.degree_file = self.datasetpath + '_graphe.deg'
        self.degree_table = []

        # key is No. node, value is its neighbour nodes
        self.nodes_dict = {}

        self.graph_in_memory = []

        # key is degree, value is number of nodes
        self.degree_distribution = {}
        self.degree_distribution_file = self.datasetpath + '_graphe.dn'

        self.cum_degree_distribution = {}

        # initalization
        self.compute_node_number()
        self.compute_node_degree()
        self.compute_nodes_dict()

    # exercise_2 : compute the number of nodes
    #@run_time
    def compute_node_number(self):
        l = []
        for element in self.dataset:
            for i in element:
                l.append(i)
        with open(self.node_file,'w') as fn:
            fn.write(str(max(l)))
        self.node_number = max(l)

    # exercise_3 : compute the degree of each node
    #@run_time
    def compute_node_degree(self):
        self.degree_table = [0] * (self.node_number + 1)
        for element in self.dataset:
            for i in element:
                if self.degree_table[i] != 0:
                    self.degree_table[i] += 1
                else:
                    self.degree_table[i] = 1
        with open(self.degree_file, 'w') as fd:
            for n in self.degree_table:
                fd.write('%s\n' % n)

    # exercise_4 : store in the memory
    #@run_time
    def store_in_memory(self):
        self.graph_in_memory = [0] * sum(self.degree_table)
        # build the index table for storage table
        index_table =[]
        index_base = 0
        for i in self.degree_table:
            index_table.append(index_base)
            index_base += i
        # index of i and j
        for element in self.dataset:
            i, j = element
            # the index of certain node in storage table
            index_i, index_j = index_table[i], index_table[j]
            # add the node to each other's table
            self.graph_in_memory[index_i] = j
            index_table[i] += 1
            self.graph_in_memory[index_j] = i
            index_table[j] += 1
        return self.graph_in_memory

    # exercise_5 : compute some infos about graph
    #@run_time
    def graph_infos(self):
        print 'Numbers of degree 0: %s' % self.degree_table.count(0)
        print 'Max Degree: %s' % max(self.degree_table)
        print 'Min Degree: %s' % min(self.degree_table)
        print 'Average Degree: %s' % (sum(self.degree_table) * 1.0 / len(self.degree_table))

    # exercise_6 : compute the degree distrubition
    #@run_time
    def compute_degree_distribution(self):
        for degree in self.degree_table:
            if degree in self.degree_distribution:
                self.degree_distribution[degree] += 1
            else:
                self.degree_distribution[degree] = 1
        with open(self.degree_distribution_file, 'w') as f:
            for degree, node in self.degree_distribution.iteritems():
                f.write('%s %s\n' % (degree , node))

    # exercise_7 : delete loop & duplicate element like (i,j) & (j,i)
    #@run_time
    def process_dataset(self, dataset):
        with open(dataset, 'r') as f:
            l = []
            for line in f.readlines():
                i, j = [int(x) for x in line.strip().split(' ')]
                if i != j:
                    l.append([i, j] if i >j else [j, i])
            
            #filter multiple edges
            for k, g in itertools.groupby(sorted(l)):
                self.dataset.append(k)
        return self.dataset

    # exercise_8 : compute the cumlative degree distribution
    #@run_time
    def cumlative_degree_distribution(self):
        self.cum_degree_distribution = self.degree_distribution.copy()
        # link copy and cum_degree_distribution just for easily reading
        copy = self.cum_degree_distribution
        n = 0 #inital nodes = 0

        for degree in reversed(sorted(copy.keys())):
            n = copy[degree] + n # add nodes has bigger degree to previous nodes
            copy[degree] = n
        return self.cum_degree_distribution

    #@run_time
    def compute_nodes_dict(self):
        self.nodes_dict = {n:[] for n in xrange(self.node_number + 1)}
        for element in self.dataset:
            i, j = element
            self.nodes_dict[i].append(j)
            self.nodes_dict[j].append(i)
        return self.nodes_dict

    #@run_time
    def compute_all(self):
        self.store_in_memory()
        self.graph_infos()
        self.compute_degree_distribution()
        self.cumlative_degree_distribution()

    # exercise_10 : compute cluster coefficient and transitive ratio for each node
    #@run_time
    def compute_triangle_values(self):
        cc, tr = [], 0
        num_v, num_tri, sum_tri = 0, 0, 0

        for n, v in self.nodes_dict.iteritems():
            if not v:
                cc.append(0) # cc of node without connection should be undefined
            elif len(v) == 1:
                cc.append(0.0) # 1 neighbour => no value
            else:
                num_v += nCr(len(v), 2)
                for u1 in v:
                    for u2 in self.nodes_dict[u1]:
                        if u2 in v:
                            num_tri += 1
                # (u1, u2) and (u2, u1) should be same, my algorithm counts twice,so here merge them
                num_tri = num_tri / 2.0
                sum_tri += num_tri
                cc.append((2 * num_tri)/ (len(v) * (len(v) - 1) ))
                num_tri = 0
        tr = sum_tri / num_v
        #print 'clustering coefficient:', cc
        print 'transitive ratio: %0.11f' % tr
        print 'average clustering coefficient: %0.11f' % (sum(cc) / self.node_number)
        average_cc = (sum(cc) / self.node_number)
        return average_cc

if __name__  == "__main__":
    pass
    
    

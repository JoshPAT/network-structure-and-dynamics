#!/usr/bin/env python 2.7.8
# -*- coding: utf-8 -*-

'''
This program is about Part 2 'Basic operations and properties'.
'''

import functools ,time, itertools
import matplotlib.pyplot as plt

# a dectorator used to compute run time in a fucntion
def run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print "Computation time: %s in Function %s" % (end_time - start_time, func.__name__)
        return result
    return wrapper

# exercise_2
@run_time
def compute_node_number(dataset, output_n = False):
    with open(dataset, 'r') as f:
        maxn = 0
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')]
            if i > maxn:
                maxn = i
            if j > maxn:
                maxn = j                  
    if output_n:
        with open(output_n,'w') as graphe_n_file:
            graphe_n_file.write(str(maxn))
    else:
        return maxn

# exercise_3
@run_time
def compute_node_degree(dataset, graphe_n, graphe_dg = False):
    with open(graphe_n, 'r') as fn:
        n = int(fn.read())
    with open(dataset, 'r') as f:
        dg_table = {}
        for x in xrange(n+1):
            dg_table[x] = 0
        for line in f.readlines():
            for e in line.strip().split(' '):
                if dg_table[int(e)] != 0:
                    dg_table[int(e)] += 1
                else:
                    dg_table[int(e)] = 1
    if graphe_dg:
        with open(graphe_dg, 'w') as graphe_dg_file:
            for dg in dg_table.values():
                graphe_dg_file.write('%s\n' % dg)
    else:
        return dg_table

# exercise_4
@run_time
def store_in_memory(dataset, graphe_n, graphe_dg):
    dg_table = []
    with open(graphe_n, 'r') as fn:
        n = int(fn.read())
    with open(graphe_dg, 'r') as fdg:
        for line in fdg.readlines():
            dg_table.append(int(line.strip()))
    with open(dataset, 'r') as f:
        '''
        store the table in 'array list' way
        '''
        # build two empty tables
        storage_table = [0] * sum(dg_table)
        # build the index table for storage table 
        index_table =[]
        s = 0
        for x in dg_table:
            index_table.append(s)
            s += int(x)
        # index of i and j
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')] 
            # the index of certain node in storage table
            index_i, index_j = index_table[i], index_table[j]
            # add the node to each other's table
            storage_table[index_i] = j
            index_table[i] += 1
            storage_table[index_j] = i
            index_table[j] += 1
        print storage_table

# exercise_5
@run_time
def compute_all_degree(graphe_dg):
    dg_0 = 0
    l = []
    dg_density = 0
    with open(graphe_dg, 'r') as fdg:
        for line in fdg.readlines():        
            if line.strip() == '0':
                dg_0 += 1
            l.append(int(line.strip()))
        dg_max = max(l) 
        dg_min = min(l)
        dg_density = 1.0 * sum(l) / (len(l) * (len(l) - 1))
        dg_average = sum(l) * 1.0 / len(l)
    print 'Numbers of degree 0: %s' % dg_0
    print 'Max Degree: %s' % dg_max
    print 'Min Degree: %s' % dg_min
    print 'Average Degree: %s' % dg_average
    print 'Density of graph: %s' % dg_density

# exercise_6
@run_time
def compute_degree_distribution(graphe_dg, graphe_dn):
    with open(graphe_dg, 'r') as f:
        d = {}
        for line in f.readlines():
            dg = int(line.strip())
            if dg in d:
                d[dg] += 1
            else:
                d[dg] = 1
    with open(graphe_dn, 'w') as f:
        for degree, node in d.iteritems():
            f.write('%s %s\n' % (degree , node))
    return d

# exercise_7
@run_time
def del_loop(raw_dataset, prelmry_dataset):
    raw_list = []
    with open(raw_dataset, 'r') as f:
        for line in f.readlines():
            i, j = [int(x) for x in line.strip().split(' ')] 
            if i != j:
                raw_list.append([i, j] if i >j else [j, i])
    raw_list.sort()
    prelmry_list = list(raw_list for raw_list,_ in itertools.groupby(raw_list))
    with open(prelmry_dataset, 'w') as fp:
        for n in prelmry_list:
            i, j = n[0], n[1]
            fp.write('%s %s\n' % (i, j))

# exercise_8
@run_time
def cumlative_degree_distribution(dg_dict):
    c = 0
    dg_c = dg_dict.copy()
    for d in sorted(dg_c.keys()):
        c = dg_c[d] + c
        dg_c[d] = c
    return dg_c

# a combination work of exercise 3 - 7
@run_time
def compute_all(dataset, file_n, file_dg, file_dn):
    compute_node_number(dataset, file_n)
    compute_node_degree(dataset, file_n, file_dg)
    store_in_memory(dataset, file_n, file_dg)
    compute_all_degree(file_dg)
    dg = compute_degree_distribution(file_dg, file_dn)
    dg_c = cumlative_degree_distribution(dg)
    return [dg, dg_c]

def make_plot(plot1, plot2):
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.scatter(plot1.keys(), plot1.values())
    ax1.axis([1, 1000, 1, 10000])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.scatter(plot2.keys(), plot2.values())
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.show()

if __name__  == "__main__":
    # test files
    dataset = 'dataset.txt'
    graphe_n = 'graphe.n'
    graphe_dg = 'graphe.dg'
    graphe_dn = 'graphe.dn'
    processed_dataset = 'processed_dataset.txt'
    # exercise 2
    compute_node_number(dataset, graphe_n)
    # exercise 3
    compute_node_degree(dataset, graphe_n, graphe_dg)
    # exercise 4
    store_in_memory(dataset, graphe_n, graphe_dg)
    # exercise 5
    compute_all_degree(graphe_dg)
    # exercise 6
    compute_degree_distribution(graphe_dg, graphe_dn)
    # exercise 7
    del_loop(dataset, processed_dataset) # this is my own dataset

    # dataset from drosophila_PPI.txt
    dataset_sophia = 'drosophila_PPI.txt'
    processed_dataset_sophia = 'processed_dataset_sophia.txt'
    del_loop(dataset_sophia, processed_dataset_sophia) # drosophila dataset
    n_s = 'sophia_graphe.n'
    dg_s = 'sophia_graphe.dg'
    dn_s = 'sophia_graphe.dn'
    
    d_s, d_s_c = compute_all(processed_dataset_sophia, n_s, dg_s, dn_s)
    make_plot(d_s, d_s_c)

    # dataset from inet.txt
    dataset_inet = 'inet.txt'
    processed_dataset_inet = 'processed_dataset_inet.txt'
    del_loop(dataset_inet, processed_dataset_inet)
    n_i = 'inet_graphe.n'
    dg_i = 'inet_graphe.dg'
    dn_i = 'inet_graphe.dn'
    
    d_i, d_i_c = compute_all(processed_dataset_inet, n_i, dg_i, dn_i)
    make_plot(d_i, d_i_c)
        

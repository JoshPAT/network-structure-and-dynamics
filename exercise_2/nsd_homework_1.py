#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Josh zhou'

#!/usr/bin/env python 2.7.8
# -*- coding: utf-8 -*-

'''
This program is about Part 2 'Basic operations and properties'.
'''

import functools ,time, itertools, operator, math, datasets
import matplotlib.pyplot as plt

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
        print self.graph_in_memory
        return self.graph_in_memory

    # exercise_5 : compute some infos about graph
    #@run_time
    def graph_infos(self):
        print 'Numbers of degree 0: %s' % self.degree_table.count(0)
        print 'Max Degree: %s' % max(self.degree_table)
        print 'Min Degree: %s' % min(self.degree_table)
        print 'Average Degree: %s' % (sum(self.degree_table) * 1.0 / len(self.degree_table))
        print 'Density of graph: %s' % (1.0 * sum(self.degree_table) / (len(self.degree_table) * (len(self.degree_table) - 1)))

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
    def make_plot(self):
        plot1, plot2 = self.degree_distribution, self.cum_degree_distribution
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey = True, figsize=(14, 6))
        ax1.scatter(plot1.keys(), plot1.values())
        ax1.axis([1, 1000, 1, 10000])
        ax1.set_xscale('log')
        ax1.set_xlabel('Number of nodes', fontsize = 14)
        ax1.set_yscale('log')
        ax1.set_ylabel('Degree', fontsize = 14)
        ax1.set_title('Degree Distribution')
        ax2.scatter(plot2.keys(), plot2.values())
        ax2.set_xlabel('Number of nodes', fontsize = 14)
        ax2.set_ylabel('Degree', fontsize = 14)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Cumulative Degree Distribution')
        plt.show()

    #@run_time
    def compute_all(self):
        self.compute_node_number()
        print self.node_number
        self.compute_node_degree()
        print self.degree_table
        self.store_in_memory()
        self.graph_infos()
        self.compute_degree_distribution()
        print self.degree_distribution
        self.cumlative_degree_distribution()
        print self.cum_degree_distribution
        self.compute_nodes_dict()
        print self.nodes_dict

    # exercise_10 : compute cluster coefficient and transitive ratio for each node
    @run_time
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
        print 'clustering coefficient:', cc
        print 'transitive ratio: %0.11f' % tr
        return cc

def distribution(li):
    distribution = {}
    for key in li.values():
            if key in distribution:
                distribution[key] += 1
            else:
                distribution[key] = 1
    return distribution

def make_plot_cc(density_distribution_1, density_distribution_2):
    plot1, plot2 = density_distribution_1, density_distribution_2
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey = True, figsize=(14, 6))
    ax1.scatter(plot1.values(), plot1.keys())
    ax1.axis([0, 50, 0, 1])
    ax1.set_xlabel('Number of nodes', fontsize = 14)
    ax1.set_ylabel('Clustering coefficient', fontsize = 14)
    ax1.set_title('Sophia')
    ax2.scatter(plot2.values(), plot2.keys())
    ax2.set_xlabel('Number of nodes', fontsize = 14)
    ax2.set_ylabel('Clustering coefficient', fontsize = 14)
    ax2.set_title('Inet')
    plt.show()


class Bfs(object):
    '''
    this class is used to compute shortest path route and numbers of
    the shortest path and betweenness centrality of a a graph.


    '''
    def __init__(self, graph):
        self.graph = graph
        self.biggest_component = []
        self.datasetpath = graph.datasetpath

    # exercise_11 : show the BFS of graph
    #@run_time
    def bfs(self, s):
        discovered = {s: 0} #mark(s)
        level = [s] #enqueue(s) -> level
        while len(level) > 0: #while level not empty:
            next_level = []
            for u1 in level:
                for u2 in self.graph.nodes_dict[u1]:
                    if u2 not in discovered:
                        discovered[u2] = u1
                        next_level.append(u2)
                        print '%d => %d' %(u1,u2),
            level = next_level #

        #sorted_discovered = sorted(discovered.items(), key = operator.itemgetter(1))
        #print ['(%d, %d)  ' % (v, k ) for v, k in sorted_discovered]
        #return discovered

    # exercise_12 : compute Size of the connected components
    #@run_time
    def compute_size(self):
        discovered = [None] * len(self.graph.nodes_dict.keys())
        for x,_ in enumerate(discovered):
            if discovered[x] is None:
                if not self.graph.nodes_dict[x]:
                    discovered[x] = 'connectless'
                else:
                    level = [x]
                    while len(level) > 0:
                        next_level = []
                        for u1 in level:
                            for u2 in self.graph.nodes_dict[u1]:
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
        d = {}
        for size in sorted(component.values()):
            if size in d:
                d[size] += 1
            else:
                d[size] = 1

        plt.scatter(d.keys(), d.values())
        plt.axis([1, 100, 1, 100000])
        plt.xscale('log')
        plt.ylabel('Size')
        plt.yscale('log')
        plt.xlabel('Number of the component')
        plt.title('Distribution of the Component Size')
        plt.show()

        # exercies_13 : isloates the biggest componet
        root = sorted_componet[0][0]
        size = sorted_componet[0][1]
        print
        print 'biggest component: %d, size: %d' % (root, size)
        biggest_component = []
        for x,_ in enumerate(discovered):
            if discovered[x] == root:
                biggest_component.append(x)
        self.biggest_component = biggest_component
        return self.biggest_component

    # exercise_14 : returns the dag of a graph
    #@run_time
    def set_of_shortest_paths(self, s):
        dag = {s: []}
        flow = dict.fromkeys(self.graph.nodes_dict, 0.0)
        flow[s] = 1.0
        stage_table = [0] * (self.graph.node_number + 1)
        level = [s]
        while len(level) > 0:
            next_level = []
            for u1 in level:
                flowu = flow[u1]
                for u2 in self.graph.nodes_dict[u1]:
                    if u2 not in dag:
                        stage_table[u2] = stage_table[u1] + 1
                        dag[u1].append(u2)
                        dag[u2] = []
                        next_level.append(u2)
                    if stage_table[u2] == stage_table[u1] + 1:
                        flow[u2] += flowu #count the flow
                        if u2 not in dag[u1]:
                            dag[u1].append(u2)
            level = next_level
        return dag


    # exercise_15 : returns the number of shortest paths go through v from s
    #@run_time
    def number_of_shortest_paths(self, s, v):
        dag = {s: []}
        flow = dict.fromkeys(self.graph.nodes_dict, 0.0)
        flow[s] = 1.0
        stage_table = [0] * (self.graph.node_number + 1)
        level = [s]
        while len(level) > 0:
            next_level = []
            for u1 in level:
                flowu = flow[u1]
                for u2 in self.graph.nodes_dict[u1]:
                    if u2 not in dag:
                        stage_table[u2] = stage_table[u1] + 1
                        dag[u1].append(u2)
                        dag[u2] = []
                        next_level.append(u2)
                    if stage_table[u2] == stage_table[u1] + 1:
                        flow[u2] += flowu #count the flow
                        if u2 not in dag[u1]:
                            dag[u1].append(u2)
            level = next_level
        child_level = dag[v]
        child_path = 0
        while len(child_level) > 0:
            next_level = []
            for child in child_level:
                child_path += 1
                next_level += dag[child]
            child_level = next_level
        print child_path * flow[v] #dn * up

    # exercise_16 : returns betweenness centrality of one node
    #@run_time
    def betweenness_centrality(self):
        bc = {x: [] for x in dict.fromkeys(self.graph.nodes_dict, 0.0)}
        for s in dict.fromkeys(self.graph.nodes_dict):
                dag = {s: []}
                flow = dict.fromkeys(self.graph.nodes_dict, 0.0)
                flow[s] = 1.0
                stage_table = [0] * (self.graph.node_number + 1)
                level = [s]
                while len(level) > 0:
                    next_level = []
                    for u1 in level:
                        flowu = flow[u1]
                        for u2 in self.graph.nodes_dict[u1]:
                            if u2 not in dag:
                                stage_table[u2] = stage_table[u1] + 1
                                dag[u1].append(u2)
                                dag[u2] = []
                                next_level.append(u2)
                            if stage_table[u2] == stage_table[u1] + 1:
                                flow[u2] += flowu #count the flow
                                if u2 not in dag[u1]:
                                    dag[u1].append(u2)
                    level = next_level
                for v in dict.fromkeys(self.graph.nodes_dict):
                    if v != s:
                        child_level = dag[v]
                        sigma = {v: [] for v in dict.fromkeys(self.graph.nodes_dict)}
                        while len(child_level) > 0:
                            next_level = []
                            for child in child_level:
                                next_level += dag[child]
                                sigma[v].append(flow[v] / flow[child])
                            child_level = next_level
                        bc[v].append(sum(sigma[v]))
                        print sum(sigma[v])
                print bc[v] #s -> t and t -> s counted twice

    @run_time
    def new_betweenness_centrality(self):
        bc = dict.fromkeys(self.graph.nodes_dict, 0.0)
        for s in self.graph.nodes_dict.keys():
            stack = []
            parent = {x: [] for x in self.graph.nodes_dict.keys()}
            sigma = dict.fromkeys(self.graph.nodes_dict, 0.0)
            sigma[s] = 1.0
            distance = dict.fromkeys(self.graph.nodes_dict, -1)
            distance[s] = 0
            queue = [s]
            while len(queue) > 0:
                v = queue.pop(0)
                stack.append(v)
                for neighour in self.graph.nodes_dict[v]:
                    if distance[neighour] < 0:
                        queue.append(neighour)
                        distance[neighour] = distance[v] + 1
                    if distance[neighour] == distance[v] + 1:
                        sigma[neighour] = sigma[neighour] + sigma[v]
                        parent[neighour].append(v)
            delta = dict.fromkeys(self.graph.nodes_dict, 0.0)
            while len(stack) > 0:
                w = stack.pop()
                for v in parent[w]:
                    delta[v] += (sigma[v] * (1.0 + delta[w]) / sigma[w])
                if w != s: bc[w] +=  delta[w]
        for k,v in bc.iteritems():
            bc[k] = v/ 2.0
        with open(self.datasetpath + '_bc.dn', 'w') as f:
            for k,v in bc.iteritems():
                f.write('%s %s\n' % (k, v))
        return bc


if __name__  == "__main__":
    g= Graph('dataset.txt')
    g.compute_all()
    #gs = Graph('drosophila_PPI.txt')
    #gs.compute_all()
    #gs.make_plot()
    #gs.compute_triangle_values()
    #gi = Graph('inet.txt')
    #gi.compute_all()
    #gi.make_plot()
    #gi.compute_triangle_values()
    #c1, c2 = distribution_cc(gs.compute_triangle_values()), distribution_cc(gi.compute_triangle_values())
    #make_plot_cc(c1, c2)

    b = Bfs(g)
    #print b.bfs(3)
    #print b.set_of_shortest_paths(3)
    #b.number_of_shortest_paths(3, 1)
    #b.betweenness_centrality(0)
    b.betweenness_centrality()
    cb = b.new_betweenness_centrality()
    d = distribution(cb)
    print d
    plt.scatter(d.values(), d.keys())
    #plt.axis([1, 100, 1, 1000])
    #plt.xscale('log')
    plt.ylabel('betweenness centrality')
    #plt.yscale('log')
    plt.xlabel('Number of the nodes')
    #plt.title('Distribution of the Component Size')
    plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

from graph import Graph, run_time
import datasets, itertools, operator
from collections import defaultdict

class Feature(Graph):
    '''
    this class is used to compute shortest path route and numbers of
    the shortest path and betweenness centrality of a a graph.

    '''
    def __init__(self, dataset = None):
        super(Feature, self).__init__(dataset)
        self.biggest_component = []
        self.compute_all()

    # compute Size of the connected components
    #@run_time
    def compute_size(self):
        discovered = [None] * len(self.nodes_dict.keys())
        for x,_ in enumerate(discovered):
            if discovered[x] is None:
                if not self.nodes_dict[x]:
                    discovered[x] = 'connectless'
                else:
                    level = [x]
                    while len(level) > 0:
                        next_level = []
                        for u1 in level:
                            for u2 in self.nodes_dict[u1]:
                                if discovered[u2] is None:
                                    discovered[u2] = x
                                    next_level.append(u2)
                        level = next_level
        component = {}
        for k, g in itertools.groupby(sorted(discovered)):
            #if k != 'connectless':
            size = len(list(g))
            component[k] = size
            print 'component: %s, size: %s' % (k, size)

        sorted_componet = sorted(component.items(), key = operator.itemgetter(0))
        
        # exercies_13 : isloates the biggest componet
        root = sorted_componet[0][0]
        size = sorted_componet[0][1]
        print 'biggest component: %d, size: %d' % (root, size)
        biggest_component = []
        for x,_ in enumerate(discovered):
            if discovered[x] == root:
                biggest_component.append(x)
        self.biggest_component = biggest_component
        return self.biggest_component

    #@run_time
    def average_distance(self):
        all_distance = 0
        '''
        use approximation to compute the average distance
        take the advantage of the property of the graph
        '''
        num = len(self.biggest_component)
        for s in self.biggest_component:
            stack = []
            distance = {k: -1 for k in self.biggest_component}
            distance[s] = 0
            queue = [s]
            added = 0
            while len(queue) > 0:
                v = queue.pop(0)
                stack.append(v)
                for neighour in self.nodes_dict[v]:
                    if distance[neighour] < 0:
                        queue.append(neighour)
                        distance[neighour] = distance[v] + 1
                        added += distance[neighour]
            all_distance += added * 1.0 / (num - 1)
        print 'average distance: %0.11f' % (all_distance * 1.0 / num)

if __name__ == '__main__':
    import models
    m = models.model_ER()
    f = Feature(m.file)
    f.compute_triangle_values()
    f.compute_size()
    f.average_distance()


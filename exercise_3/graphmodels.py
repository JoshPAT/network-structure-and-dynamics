#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Josh zhou'

import random, numpy as np, functools, time
from collections import defaultdict

def run_time(func):
	@functools.wraps(func)
	def wrapper(*args, **kw):
		start_time = time.time()
		result = func(*args, **kw)
		end_time = time.time()
		print "Computation Time of %s: %s" % (func.__name__.capitalize(), end_time - start_time)
		return result
	return wrapper

class ErdosRenyiModels(object):
	def __init__(self):
		self.file = 'erdos_renyi.txt'

	# exercise_1 Generation
	def generation(self, n = 0, m = 0):
		with open('datasets/' + self.file, 'w') as f:
			graph = [0] * n
			for _ in xrange(m):
				#random.seed()
				i = random.randint(0, n-1)
				j = random.randint(0, n-1)
				while i == j:
					j = random.randint(0, n-1)
				f.write('%i %i\n' % (i, j))
				graph[i] = j
		return graph

class RandomFixedDegreeModels(object):
	def __init__(self):
		self.file = 'fixed_table.txt'

	def direct_generation(self, degree_table = None):
		i , fixed_table = 0, []
		with open('datesets' + degree_table, 'r') as f:
			for line in f.readlines():
				d = int(line.strip())
				for _ in xrange(d):
					fixed_table.append(i)
				i += 1
		with open('datasets/' + self.file, 'w') as f:
			i = len(fixed_table)
			while i > 0:
				#random.seed()
				u = random.randint(0, i - 1)
				fixed_table[i-1], fixed_table[u] = fixed_table[u], fixed_table[i-1]
				v = random.randint(0, i - 2)
				fixed_table[i-2], fixed_table[v] = fixed_table[v], fixed_table[i-2]
				i = i - 2
				f.write('%d %d\n' % (u, v))

	@run_time
	def switch_generation(self, dataset):		
		vector= []
		with open('datasets/' + dataset, 'r') as f:
			for line in f.readlines():
				vector.append([int(e) for e in line.strip().split(' ')])
		vector = np.array(vector)
		edges = defaultdict(list)
		for i, j in vector:
			edges[i].append(j)
			edges[j].append(i)
		for _ in xrange(10 ** 6):
			i = np.random.randint(0, len(vector) - 1) # row i
			j = np.random.randint(0, len(vector) - 1) # row j
			while vector[i, 0] == vector[j,0] : j = np.random.randint(0, len(vector) - 1) # make sure [i][0] != [j][0] 
			# now do the switch end
			vector[i, 1], vector[j, 1] = vector[j, 1], vector[i, 1]
			while True:
				# aviod loops
				if vector[i, 0] != vector[j, 1]:
					if vector[j, 0] != vector[i, 1]:
						#check if any multiple edge	
						if vector[i, 1] not in edges[j]:
							if vector[j, 1] not in edges[j]:
								break
				j = np.random.randint(0, len(vector) - 1)
				#if len([tuple(row) for row in vector]) == len(vector): break  #this method is very low
		
		with open('datasets/' + self.file, 'w') as f:
			for u, v in vector:
				f.write('%d %d\n' % (u, v))
				

if __name__== '__main__':
	import random, os
	#m = ErdosRenyiModels()
	#exercise_2 Characterists
	#model = m.generation(7236, 22270)
	#print m.generation()
	m = RandomFixedDegreeModels()
	#m.generation('drosophila_PPI_graphe.deg')
	m.switch_generation('drosophila_PPI.txt')
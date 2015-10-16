#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import random, numpy as np, functools, time
from collections import defaultdict
from graph import Graph

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
	def generate(self, n = 0, m = 0):
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
	def __init__(self, option = 'direct'):
		if option == 'direct':
			self.file = 'direct_fixed_table.txt'
		elif option == 'switch':
			self.file = 'switch_fixed_table.txt'
			self.permutation_file = 'permutation_file.txt'
			self.cluster_file = 'cluster_file.txt'

	def direct_generate(self, degree_table = None):
		n , fixed_table = 0, []
		with open('datasets/' + degree_table, 'r') as f:
			for line in f.readlines():
				d = int(line.strip())
				for _ in xrange(d):
					fixed_table.append(n)
				n += 1
		with open('datasets/' + self.file, 'w') as f:
			i = len(fixed_table)
			while i > 0:
				#random.seed()
				while True:
					u = random.randint(0, i - 1)
					v = random.randint(0, i - 2)
					if fixed_table[u] != fixed_table[v]:
						break
				fixed_table[i-2], fixed_table[v] = fixed_table[v], fixed_table[i-2]
				fixed_table[i-1], fixed_table[u] = fixed_table[u], fixed_table[i-1]
				f.write('%d %d\n' % (fixed_table[i-1], fixed_table[i-2]))
				i = i - 2

	@run_time
	def switch_generate(self, dataset, t = 10 ** 6, recompute = False):		
		if recompute:
			with open('datasets/' + self.cluster_file, 'w') as f:
					pass # initalization
		vector= []
		with open('datasets/' + dataset, 'r') as f:
			for line in f.readlines():
				i, j = [int(e) for e in line.strip().split(' ')]
				if i != j:
					vector.append([i,j])
		vector = np.array(vector)
		edges = defaultdict(list)
		
		for i, j in vector:
			edges[i].append(j)
			edges[j].append(i)
		
		for _ in xrange(t):
			# compute the cc every 10^4 times, and write it into the file 
			if recompute:
				if _ < 10 **3:
					if _ % (10 ** 2) == 0:
						with open('datasets/' + self.permutation_file, 'w') as f:
							for u, v in vector:
								f.write('%d %d\n' % (u, v))
						g = Graph(self.permutation_file)
						with open('datasets/' + self.cluster_file, 'a') as f:
							f.write('%d %0.11f\n' % (_, g.compute_triangle_values()))
				if _ < 10 **4:
					if _ % (10 ** 3) == 0:
						with open('datasets/' + self.permutation_file, 'w') as f:
							for u, v in vector:
								f.write('%d %d\n' % (u, v))
						g = Graph(self.permutation_file)
						with open('datasets/' + self.cluster_file, 'a') as f:
							f.write('%d %0.11f\n' % (_, g.compute_triangle_values()))
				else:
					if _ % (10 ** 5) == 0:
						with open('datasets/' + self.permutation_file, 'w') as f:
							for u, v in vector:
								f.write('%d %d\n' % (u, v))
						g = Graph(self.permutation_file)
						with open('datasets/' + self.cluster_file, 'a') as f:
							f.write('%d %0.11f\n' % (_, g.compute_triangle_values()))
			while True:
				r1 = np.random.randint(0, len(vector) - 1) # row 1
				i = vector[r1, 0]
				r2 = np.random.randint(0, len(vector) - 1) # retry another row 2
				j = vector[r2, 0]
				# aviod loops
				if vector[r1, 0] != vector[r2, 1]:
					if vector[r2, 0] != vector[r1, 1]:
						#check if any multiple edge	
						if vector[r1, 1] not in edges[j]:
							if vector[r2, 1] not in edges[i]:
								break
			# now do the switch end
			a, b = vector[r1]
			c, d = vector[r2]

			# remove the link
			edges[a].remove(b)
			edges[b].remove(a)
			edges[c].remove(d)
			edges[d].remove(c)
			# add the link
			edges[a].append(d)
			edges[b].append(c)
			edges[c].append(b)
			edges[d].append(a)
			
			# switch
			vector[r1, 1], vector[r2, 1] = vector[r2, 1], vector[r1, 1]

			if vector[r1,0] == vector[r1,1]:
				print vector[r1]
			if vector[r2,0] == vector[r2,1]:
				print vector[r2]
			#if len([tuple(row) for row in vector]) == len(vector): break  #this method is very low
		
		with open('datasets/' + self.file, 'w') as f:
			for u, v in vector:
				f.write('%d %d\n' % (u, v))

class BarabasiAlertModels(object):
	def __init__(self):
		self.file = 'barabasi_alert.txt'
		self.initial_graph = 'initial_graph.txt'

	@run_time
	def generate(self,n = 0, m = 0):
		'''
		n : all the nodes in random graph.
		m : number of the fixed half links for new added node.
		'''
		degree_table = np.indices((n + 1, 2))[0] # create a vector with index of the row
		degree_table[: ,1] = 0 # define all the degree of the node to 0
		maxi = 0
		with open('datasets/' + self.initial_graph, 'r') as f:
			with open('datasets/' + self.file, 'w') as f2:
				for line in f.readlines():
					f2.write(line.strip() + '\n')
					for i in line.strip().split(' '):
						degree_table[int(i), 1] += 1
						if int(i) > maxi:
							maxi = int(i)
		#print degree_table[:,0] # node
		#print np.sum(degree_table[:,1]) # degree sum
		#print degree_table
		
		# start algorithm
		with open('datasets/' + self.file, 'a') as f:
			i = maxi + 1 # start from next node of the inital node numbers -> 
			while i <= n:
				new_nodes = np.empty([m], dtype = np.int16)
				start_m = m
				for x in xrange(m):
					while 1:
						e = np.random.choice(degree_table[:,0], p = 1.0 * degree_table[:,1] / np.sum(degree_table[:,1]))
						#if e not in new_nodes:
						new_nodes[x] = e
						break
				for e in new_nodes:
					degree_table[e, 1] += 1
					f.write('%d %d\n' % (i, e))
				degree_table[i, 1] = m
				i += 1

class WattStrogatzModels(object):
	def __init__(self):
		self.inital_file = 'watt_strogatz.txt'
		self.file = 'watt_strogatz_randomized.txt'

	def generate(self, n=0, k=0, p=0):
		'''
		n: number of nodes.
		k: the mean degree of the graph.
		b: the parameter to describe the level of randomness(the bigger parameter more random the graph)
		'''
		# creat a watts-strogatz graph (b = 0)
 		vector = [] # vector stores the graph in (i, j) way
 		
 		with open('datasets/' + self.inital_file, 'w') as f:
			i = 0
			while i < (n - k / 2):
				for x in xrange(1, k/2 + 1):
					f.write('%d %d\n' % (i, i + x))
					vector.append((i, i+ x))
				i += 1
			# connect the whole graph, this is a faster way
			while i < n:
				for x in xrange(1, k/2 + 1):
					if i + x >= n:
						f.write('%d %d\n' % (i,  i + x - n))
						vector.append((i, i+ x - n))
					else:
						f.write('%d %d\n' % (i,  i + x))
						vector.append((i, i+ x ))
				i += 1
		vector = np.array(vector)
		edges = defaultdict(list)
		for i, j in vector:
			edges[i].append(j)
			edges[j].append(i)

		# for each row1 in the vector
		for r1 in xrange(len(vector)):
			if np.random.ranf() <= p:
				while True:
					r2 = np.random.randint(0, len(vector) - 1) # row 2
					i = vector[r1, 0]
					j = vector[r2, 0]
					# aviod loops
					if vector[r1, 0] != vector[r2, 0]:
						#check if any multiple edge	
						if vector[r1, 1] not in edges[j]:
							break

				# now change the link
				a, b = vector[r1]
				c, d = vector[r2]
				# remove the link
				edges[a].remove(b)
				edges[b].remove(a)
				# add the link
				edges[c].append(a)
				edges[a].append(c)
				# change
				vector[r1, 1] = vector[r2, 0]

		with open('datasets/' + self.file, 'w') as f:
			for u, v in vector:
				f.write('%d %d\n' % (u, v))

def model_ER():
	m = ErdosRenyiModels()
	m.generate(7236, 22270)
	return m

def model_FD(option = 'direct', r = False):
	m = RandomFixedDegreeModels(option)
	if option == 'switch':
		m.switch_generate('drosophila_PPI.txt', recompute = r)
	elif option == 'direct':
		m.direct_generate('drosophila_PPI_graphe.deg')
	return m

def model_BA():
	m = BarabasiAlertModels()
	m.generate(7235, 6)
	return m

def model_WS(n,k,p):
	m = WattStrogatzModels()
	m.generate(n,k,p)
	return m

if __name__== '__main__':
	import random
	



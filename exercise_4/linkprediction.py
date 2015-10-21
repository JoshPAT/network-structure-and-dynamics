#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import os
import numpy as np
import argparse
import logging
import functools
from graph import run_time
from collections import *

# A Fast Way to create a dir
def mkdir(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        # Create a dir
        global dirpath
        dir_name = func.__name__
        dirpath = os.path.join('datasets/', dir_name)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        result = func(*args, **kw)
        return result
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        nargs='?',
        type=argparse.FileType('r'),
        help="Select the dataset your want to create the file. \
                Default is 'drosophila_PPI.txt'",
        default="datasets/drosophila_PPI.txt"
    )
    parser.add_argument(
        "-n",
        "--number",
        nargs="?",
        type=int,
        help="Give the number of missing nodes. \
                Default is '2000'",
        default=2000
    )
    parser.add_argument(
        "-r",
        "--reset",
        action="store_true",
        help="Recompute the missed links and restore the files."
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        nargs="?",
        help="Execute the Link Prediction method. \n \
              'j' stands for jaccard ranking. \
              ",
        default=None
    )
    return parser.parse_args()


@run_time
def data_preparation(args):
    '''
    Select randomly 2000 links of the dataset,
    Store them in the 'datasets' file.
    '''
    global nodes_number
    global num_missed_links
    f = args.dataset
    num_missed_links = args.number
    reset = args.reset

    # create a martix contains all the links
    orginal_links = np.array(
        [line.strip().split(' ') for line in f.readlines()],
        dtype = np.int16
    )
    
    # To have the global number of the nodes
    nodes_number = np.amax(orginal_links)

    if not os.path.exists('datasets/missed_links.txt') or \
        not os.path.exists('datasets/sample_links.txt') or \
        reset:
        if reset:
            logging.info('Reset files. Start to Create...')
        else:
            logging.info('No files. Start to Create...')
        
        # This is much faster way to get the data, not check one by one
        randlist = np.random.randint(len(orginal_links), size=num_missed_links + num_missed_links/4)
        randlist = np.unique(randlist)[:num_missed_links]
        missed_links = orginal_links[randlist]
        sample_links = np.delete(orginal_links, randlist, axis=0)
        
        # To Save both the sample links and missed links
        filepath = os.path.split(os.path.abspath(__file__))[0]

        with open(os.path.join(filepath, 'datasets/missed_links.txt'),
                     'w') as file_1:
            for row in missed_links:
                file_1.write('%d %d\n' % (row[0], row[1]))
            logging.info('Successfully generate the missed links file')

        with open(os.path.join(filepath, 'datasets/sample_links.txt'),
                     'w') as file_2:
            for row in sample_links:
                file_2.write('%d %d\n' % (row[0], row[1]))
            logging.info('Successfully generate the sample links file')


class Local_Scoring():
    @staticmethod
    @mkdir
    def jaccard_ranking():
        '''
        In numpy, it enables to sort array by certain name:
        
        j_links = np.array(
                    j_links,  # have to be tuple-like array
                    dtype = [
                        ('i', 'i2'),
                        ('j', 'i2'),
                        ('s', 'f8')
                        ]
                    )
        j_links = np.sort(j_links, order = 's')[:-num_missed_links]

        However, pypy interper doesn't fully support this function.

        Time using the pypy interpreter - Jaccard_ranking: 48.8150649071
        Time using the python interpreter - Jaccard_ranking: 68.9379351139
        '''
        result_file = os.path.join(dirpath, 'results.txt')

        if not os.path.exists(result_file):

            logging.info('Start to Compute...')
            # Read the sample file
            with open('datasets/sample_links.txt', 'r') as f:
                sample_links = np.array(
                    [line.strip().split(' ') for line in f.readlines()],
                    dtype = np.int16
                )
                
            # Create set of each nodes
            d = defaultdict(set)
            for i,j in sample_links:
                if i!= j:
                    d[i].add(j)
                    d[j].add(i)

            j_links = []
            for i in xrange(nodes_number+1):
                for j in xrange(i, nodes_number+1):
                    if i != j and j not in d[i]:
                        # Union = d[i] | d[j]
                        intersection = d[i] & d[j]
                        if intersection:
                            scores = len(intersection) * 1.0 / len(d[i] | d[j])
                            j_links.append([i,j,scores])      
            
            # Get the links ranked nth
            j_links = np.array(j_links)
            ranks = np.argsort(j_links[:,2], kind = 'heapsort')
            j_links = j_links[ranks][-num_missed_links:]
            logging.info('Finished.') 

            with open(result_file, 'w') as f:
                for i, j, s in j_links:
                    f.write('%d %d %f\n' % (i, j, s))

    @staticmethod
    def adamic_adar_ranking():
        pass

    @staticmethod
    def resource_allocation_ranking():
        pass

class Global_Scoring():
    @staticmethod
    def random_paths_scoring():
        pass

    @staticmethod
    def karz_method():
        pass

class Consensus_Method():
    @staticmethod
    def Bordas_method():
        pass

def precision_recall_plots():
    pass



if __name__ == '__main__':
    # Set the log level
    logging.basicConfig(level=logging.INFO)
    # Parse args
    args = parse_args()
    # Generate the data
    data_preparation(args)
    if args.method:
        if args.method =='j':Local_Scoring.jaccard_ranking()


    

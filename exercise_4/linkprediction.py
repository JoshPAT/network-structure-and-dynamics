#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import os
import numpy as np
import argparse
from graph import run_time

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
    return parser.parse_args()


def data_preparation(args):
    '''
    Select randomly 2000 links of the dataset,
    Store it in the file.
    '''
    
    f = args.dataset
    n = args.number
    reset = args.reset

    # create a martix contains all the links
    orginal_links = np.array(
        [line.strip().split(' ') for line in f.readlines()],
        dtype = np.int16
    )
    # This is much faster way to get the data, not check one by one
    randlist = np.random.randint(len(orginal_links), size=n + n/4)
    randlist = np.unique(randlist)[:2000]
    missed_link = orginal_links[randlist]
    rest_links = np.delete(orginal_links, randlist, axis=0)
    
    if reset:
        filepath = os.path.split(os.path.abspath(__file__))[0]

        with open(os.path.join(filepath, 'datasets/missed_link.txt'),
                     'w') as file_1:
            for row in missed_link:
                file_1.write('%d %d\n' % (row[0], row[1]))

        with open(os.path.join(filepath, 'datasets/rest_links.txt'),
                     'w') as file_2:
            for row in rest_links:
                file_2.write('%d %d\n' % (row[0], row[1]))

class Local_Scoring():
    @staticmethod
    def jaccard_ranking():
        pass

    @staticmethod
    def adamic_adar_ranking():
        pass

    @staticmethod
    def resource_allocation_ranking():
        pass

if __name__ == '__main__':
    # Parse args
    args = parse_args()
    # Generate the data
    data_preparation(args)

    

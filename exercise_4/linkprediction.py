#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import os, shutil, argparse, logging, functools, time, collections, math, itertools, random, operator
import numpy as np
from scipy.sparse import *
import plotly.plotly as py
import plotly.graph_objs as go

# a dectorator used to compute run time in a fucntion
def run_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        end_time = time.time()
        print "Computation Time of %s: %s" % \
              (func.__name__.capitalize(), end_time - start_time)
        return result
    return wrapper


# A Fast Way to create a dir
def mkdir(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        # Create a dir
        global dirpath
        dir_name = func.__name__
        dirpath = os.path.join('outputs/', dir_name)
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
        help="Execute the Link Prediction method. \
              'lj' stands for jaccard ranking.   \
              'la' stands for adamic adar ranking.  \
              '' ",
        default=None
    )
    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        nargs="?",
        help="Plot the Precision_Recall Plots.",
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

    # Reset - Delete all the files and dirs except datafile
    if reset or not os.path.exists('outputs'):
        if os.path.exists('outputs'): 
            logging.info('Deleting old files...')
            shutil.rmtree('outputs')
        os.makedirs('outputs')

        # This is much faster way to get the data, not check one by one
        randlist = np.random.randint(
            len(orginal_links), 
            size=num_missed_links + num_missed_links/4
            )
        randlist = np.unique(randlist)[:num_missed_links]
        missed_links = orginal_links[randlist]
        sample_links = np.delete(orginal_links, randlist, axis=0)
        
        # To Save both the sample links and missed links
        filepath = os.path.split(os.path.abspath(__file__))[0]

        with open(os.path.join(filepath, 'outputs/missed_links.txt'),
                     'w') as file_1:
            for row in missed_links:
                file_1.write('%d %d\n' % (row[0], row[1]))
            logging.info('Successfully generate the missed links file')

        with open(os.path.join(filepath, 'outputs/sample_links.txt'),
                     'w') as file_2:
            for row in sample_links:
                file_2.write('%d %d\n' % (row[0], row[1]))
            logging.info('Successfully generate the sample links file')

def nodes_edges(option=False):
    with open('outputs/sample_links.txt', 'r') as f:
        sample_links = np.array(
            [line.strip().split(' ') for line in f.readlines()],
            dtype = np.int16
        )
    # Create set or lists of each nodes
    if option:
        d = collections.defaultdict(list)
        for i,j in sample_links:
            if i!= j:
                d[i].append(j)
                d[j].append(i)
    else:
        d = collections.defaultdict(set)
        for i,j in sample_links:
            if i!= j:
                d[i].add(j)
                d[j].add(i)
    return d


class Local_Scoring():
    @staticmethod
    @mkdir
    @run_time
    def jaccard_ranking():
        '''
        This method is used to calculate the jaccard_ranking.

        In numpy, it enables to sort array by certain name:
        
        pair_sets = np.array(
                    pair_sets,  # have to be tuple-like array
                    dtype = [
                        ('i', 'i2'),
                        ('j', 'i2'),
                        ('s', 'f8')
                        ]
                    )
        pair_sets = np.sort(pair_sets, order = 's')[:-num_missed_links]

        However, pypy interper doesn't fully support this function.

        Time using the pypy interpreter - Jaccard_ranking: 48.8150649071
        Time using the python interpreter - Jaccard_ranking: 68.9379351139
        '''
        result_file = os.path.join(dirpath, 'results.txt')

        if not os.path.exists(result_file):
            logging.info('Start to Compute...')
            # Load the edges
            d  = nodes_edges()

            pair_sets = collections.defaultdict(float)
            for i, j in itertools.combinations(dict.fromkeys(d), 2):            
                if d[j] and d[i] and i != j:
                    # i, j are not connected
                    if j not in d[i]:
                        # Union = d[i] | d[j]
                        intersection = d[i] & d[j]
                        if intersection:                                    
                            scores = len(intersection) * 1.0 / len(d[i] | d[j])
                            pair_sets[tuple([i,j])] = scores
            

            pair_sets = sorted(
                pair_sets.items(), 
                key=operator.itemgetter(1), 
                reverse = True
                )

            with open(result_file, 'w') as f:
                for links, scores in pair_sets:
                    f.write("%d %d " % links + "%f\n" % scores)
            logging.info('Finished.')

    @staticmethod
    @mkdir
    @run_time
    def adamic_adar_ranking():
        '''
        This method is used to calculate the adamic_adar_ranking.

        '''
        result_file = os.path.join(dirpath, 'results.txt')
        
        if not os.path.exists(result_file):
            logging.info('Start to Compute...')
            # Load the edges 
            d = nodes_edges()
            pair_sets = collections.defaultdict(float)
            for i,j in itertools.combinations(dict.fromkeys(d), 2):
                if d[j] and d[i]:
                    # i, j are not connected
                    if j not in d[i]:
                        # Union = d[i] | d[j]
                        intersection = d[i] & d[j]
                        if intersection:                                    
                            scores = \
                                sum(1.0 / math.log(len(d[internode]))
                                for internode in intersection)
                            pair_sets[tuple([i,j])] = scores

            pair_sets = sorted(
                pair_sets.items(), 
                key=operator.itemgetter(1), 
                reverse = True
                )

            with open(result_file, 'w') as f:
                for links, scores in pair_sets:
                    f.write("%d %d " % links + "%f\n" % scores)
            logging.info('Finished.')

    @staticmethod
    @mkdir
    @run_time
    def resource_allocation_ranking():
        '''
        This method is used to calculate the resource_allocation_ranking.

        '''
        result_file = os.path.join(dirpath, 'results.txt')

        if not os.path.exists(result_file):
            logging.info('Start to Compute...')
            # Load the edges 
            d = nodes_edges()
            pair_sets = collections.defaultdict(float)
            for i, j in itertools.combinations(dict.fromkeys(d), 2):
                if d[j] and d[i]:
                    # i, j are not connected
                    if j not in d[i]:
                        # Union = d[i] | d[j]
                        intersection = d[i] & d[j]
                        if intersection:                                    
                            scores = \
                                sum(1.0 / len(d[internode]) \
                                for internode in intersection)
                            pair_sets[tuple([i,j])] = scores

            pair_sets = sorted(
                pair_sets.items(), 
                key=operator.itemgetter(1),
                reverse = True
                )

            with open(result_file, 'w') as f:
                for links, scores in pair_sets:
                    f.write("%d %d " % links + "%f\n" % scores)
            logging.info('Finished.')        

class Global_Scoring():
    @staticmethod
    @mkdir
    @run_time
    def random_paths_scoring():
        '''
        This method is used to generate the random path size 4.
        '''
        
        result_file = os.path.join(dirpath, 'results.txt')
        if not os.path.exists(result_file):
            logging.info('Start to generate random path...')
            d = nodes_edges(option=True)
            pair_sets = collections.defaultdict(int)
            for _ in xrange(10 ** 6):
                while 1:
                    # Random generate a node
                    node = random.randint(0,nodes_number + 1)
                    random_path = [node]
                    # Loop until there is a dead end or fullfill the condtion
                    while len(d[node]) > 1 and len(random_path) < 4:
                        next_node = random.choice(d[node])
                        # Eliminate possible path like 1-2-1-3 or 1-2-1-2
                        if next_node not in random_path:
                            random_path += [next_node]
                            node = next_node
                        else:
                            node = random.randint(0,nodes_number + 1)
                            random_path = [node]
                    # Break if fullfill the condition
                    if len(random_path) > 3:
                        break
                # Filter the existing link in the path
                for _ in xrange(2):
                    if random_path[_] not in d[random_path[_+2]]:
                        pair = tuple(sorted(random_path[_::2]))
                        pair_sets[pair] += 1
                if random_path[0] not in d[random_path[3]]:
                    pair = tuple(sorted([random_path[0], random_path[3]]))
                    pair_sets[pair] += 1

            pair_sets = sorted(
                pair_sets.items(), 
                key=operator.itemgetter(1), 
                reverse = True
                )
            with open(result_file, 'w') as f:
                for links, scores in pair_sets:
                    f.write("%d %d " % links + "%d\n" % scores)
            logging.info('Finished.')

    @staticmethod
    @mkdir
    @run_time
    def karz_method():
        '''
        This methid is used to calculate the rankings using the karz method. 
        '''
        result_file = os.path.join(dirpath, 'results.txt')
        
        if not os.path.exists(result_file):
            logging.info('Start to Compute...')

            # Make the zero array first (n x n)
            adj_matrix = np.zeros(
                shape = (nodes_number + 1, nodes_number + 1),
                dtype = np.int16
                )
            path_matrix = np.zeros(
                shape = (nodes_number + 1, nodes_number + 1),
                dtype = np.int16
                )

            # Make the adjacency matrix
            with open('outputs/sample_links.txt', 'r') as f:
                for line in f.readlines():
                    i, j = line.strip().split(' ')
                    i, j = int(i), int(j)
                    if i != j:
                        adj_matrix[i, j] += 1
                        adj_matrix[j, i] += 1
            adj_matrix = csr_matrix(adj_matrix)
            path_matrix = csr_matrix(adj_matrix)

            # SCR computes the fastest
            adj_matrix_tmp = csr_matrix.copy(adj_matrix)
            # from l = 2 to l = 4
            for l in xrange(2,5):
                # Generate the paths length equals = l
                adj_matrix_tmp = adj_matrix_tmp * adj_matrix
                path_matrix += 0.1 ** l * adj_matrix_tmp
            
            logging.info(
                'Finished Path Computation. Start to Assign the values...'
                )

            path_matrix = path_matrix.toarray()
            pair_sets = collections.defaultdict(int)            
            d = nodes_edges()
            for i,j in itertools.combinations(dict.fromkeys(d), 2):
                # i, j are not connected
                if path_matrix[i,j] != 0 and j not in d[i]:
                    pair = tuple((i, j))
                    pair_sets[pair] = path_matrix[i, j]
            
            logging.info('Finished Path Computation. Start to Sort...')

            pair_sets = sorted(
                pair_sets.items(), 
                key=operator.itemgetter(1), 
                reverse = True
                )

            with open(result_file, 'w') as f:
                for links, scores in pair_sets:
                    f.write("%d %d " % links + "%f\n" % scores)
            logging.info('Finished.')

class Consensus_Method():
    @staticmethod
    def Bordas_method():
        pass



def precision_recall_plots(dir_name):
    with open('outputs/missed_links.txt', 'r') as f:
        missed_links = np.array(
            [line.strip().split(' ') for line in f.readlines()],
            dtype = np.int16
            )
        # Sort the (i, j)
        missed_links.sort()
    
    pr = {}
    tp = 0
    test_nb = 0

    dirpath = os.path.join('outputs/', dir_name, 'results.txt')
    with open(dirpath, 'r') as f:
        predicted_links = np.array(
            [line.strip().split(' ') for line in f.readlines()]
        )
        all_number = len(predicted_links)
        f.seek(0)
        for line in f.readlines():
            test_nb += 1
            i, j, s = line.strip().split(' ')
            i, j = int(i), int(j)
            if i > j: i, j = j, i
            if any(np.equal(missed_links,[i,j]).all(1)):
                tp += 1
            if test_nb % 5000 == 0:
                pr[tp *1.0/ test_nb] = tp * 1.0/ all_number
        
        logging.info('Start to Plot...')

        plotfig = collections.OrderedDict(sorted(pr.items()))
        trace1 = go.Scatter(
            x = plotfig.values(),
            y = plotfig.keys(),
            showlegend = False,
            mode = 'lines',
        )
        data = [trace1]
        layout = go.Layout(
            title = dir_name.capitalize(),
            autosize = True,
            
            xaxis = dict(
                autorange = True,
                #title = 'Switch Times',
                #exponentformat='power',
                tickangle = 10
            ),
            yaxis = dict(

                autorange = True,
                #title = 'clustering coefficient',
                #exponentformat ='power',
                tickangle = 10
            ),
            plot_bgcolor='rgb(238, 238, 238)',
        )
        fig = go.Figure(data = data, layout =layout)
        plot_url = py.plot(fig, filename= dir_name.capitalize())


if __name__ == '__main__':
    # Set the log level
    logging.basicConfig(level=logging.INFO)
    # Parse args
    args = parse_args()
    # Generate the data
    data_preparation(args)
    if args.method:
        if args.method == 'lj':
            Local_Scoring.jaccard_ranking()
        if args.method == 'la':
            Local_Scoring.adamic_adar_ranking()
        if args.method == 'lr':
            Local_Scoring.resource_allocation_ranking()
        if args.method == 'gr':
            Global_Scoring.random_paths_scoring()
        if args.method == 'gk':
            Global_Scoring.karz_method()
        if args.method == 'cb':
            Consensus_Method.Bordas_method()
    if args.plot:
        if args.plot == 'lj':
            precision_recall_plots('jaccard_ranking')
        if args.plot == 'la':
            precision_recall_plots('adamic_adar_ranking')
        if args.plot == 'lr':
            precision_recall_plots('resource_allocation_ranking')
        if args.plot == 'gr':
            precision_recall_plots('random_paths_scoring')
        if args.plot == 'gk':
            precision_recall_plots('karz_method')
        if args.plot == 'cb':
            precision_recall_plots('Bordas_method')






    

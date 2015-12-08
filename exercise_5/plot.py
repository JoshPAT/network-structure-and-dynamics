#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import os
import argparse
import logging
import plotly.plotly as py
import plotly.graph_objs as go
from Measurement import Graph

DATASETS = {'o': 'Flickr', 't': 'Flickr-test', 'i' : 'inet' }

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
    return parser.parse_args()

class Plot(object):
    '''
    Used to plot the figures.
    '''
    def __init__(self, name):
        self.graph_name = DATASETS[name]
        self.path = os.path.join('outputs/', self.graph_name, 'strategies/')
        self.x_y = [self.find_x_y(f) for f in ['random_strategy', 'complete_strategy', 'tbf_strategy','v_random_strategy', 'combined_strategy']]
        self.calculate_efficiency()
        self.efficiency_plots()

    def find_x_y(self, f):
        x_y = {}
        find_number = 0
        test_number = 0
        with open(os.path.join(self.path, f), 'r') as f:
            for line in f.readlines():
                find_number += 1
                if find_number % 10 == 0:
                    test_number = int(line.split(' ')[0])
                    x_y[test_number] = find_number 
        return zip(*sorted(x_y.iteritems()))

    def calculate_efficiency(self):
        nb_links = os.path.join(self.path, 'links')
        with open(nb_links, 'r') as f:
            efficiency_max = int(f.read().strip())

    def efficiency_plots(self):
        logging.info('Start to Plot...')

        trace0 = go.Scatter(
            x = self.x_y[0][0],
            y = self.x_y[0][1],
            #showlegend = False,
            mode = 'lines',
            name = 'random_strategy',
        )
        trace1 = go.Scatter(
            x = self.x_y[1][0],
            y = self.x_y[1][1],
            #showlegend = False,
            mode = 'lines',
            name = 'complete_strategy',
        )
        trace2 = go.Scatter(
            x = self.x_y[2][0],
            y = self.x_y[2][1],
            #showlegend = False,
            mode = 'lines',
            name = 'tbf_strategy',
        )
        trace3 = go.Scatter(
            x = self.x_y[3][0],
            y = self.x_y[3][1],
            #showlegend = False,
            mode = 'lines',
            name = 'v_random_strategy',
        )
        trace4 = go.Scatter(
            x = self.x_y[4][0],
            y = self.x_y[4][1],
            #showlegend = False,
            mode = 'lines',
            name = 'combined_strategy',
        )

        data = [trace0, trace1, trace2, trace3, trace4]
        layout = go.Layout(
            autosize = True,
            xaxis = dict(
                autorange = True,
                #title = 'Switch Times',
                exponentformat='power',
                tickangle = 10
            ),
            yaxis = dict(
                autorange = True,
                #title = 'clustering coefficient',
                exponentformat ='power',
                tickangle = 10
            ),
            #plot_bgcolor='rgb(238, 238, 238)',
        )
        fig = go.Figure(data = data, layout =layout)
        plot_url = py.plot(fig, filename= self.graph_name.capitalize())

class Plot_Distribution(Graph):

    def __init__(self, dataname):
        super(Plot_Distribution, self).__init__(dataname) 
        self.graph_name = dataname
        self.cum_degree_distribution = self._cumlative_degree_distribution()
        self.plot_degree()

    def _cumlative_degree_distribution(self):
        self.cum_degree_distribution = self.degree_distribution.copy()
        # link copy and cum_degree_distribution just for easily reading
        copy = self.cum_degree_distribution
        n = 0 #inital nodes = 0
        for degree in reversed(sorted(copy.keys())):
            n = copy[degree] + n # add nodes has bigger degree to previous nodes
            copy[degree] = n
        return self.cum_degree_distribution

    def plot_degree(self):
        trace1 = go.Scatter(
            x = self.degree_distribution.keys(),
            y = self.degree_distribution.values(),
            mode = 'markers',
            showlegend = False
        )
        trace2 = go.Scatter(
            x = self.cum_degree_distribution.keys(),
            y = self.cum_degree_distribution.values(),
            mode ='markers',
            xaxis = 'x2',
            yaxis = 'y2',
            showlegend = False
        )
        data = [trace1, trace2]
        layout = go.Layout(
            title = self.graph_name.capitalize(),
            width = 1000,
            height = 550,
            
            xaxis = dict(
                type = 'log',
                autorange = True,
                title = 'Degree',
                exponentformat='power',
                tickangle = 10,
                domain = [0, 0.45]
            ),
            yaxis = dict(
                type = 'log',
                autorange = True,
                title = 'Number of nodes',
                exponentformat ='power',
                tickangle = 10
            ),
            xaxis2 = dict(
                type = 'log',
                autorange = True,
                title = 'Degree',
                exponentformat='power',
                tickangle = 10,
                domain = [0.55, 1]
            ),
            yaxis2 = dict(
                anchor = 'x2',
                type = 'log',
                autorange = True,
                title = 'Number of nodes',
                exponentformat ='power',
                tickangle = 10
            ),
            plot_bgcolor='rgb(238, 238, 238)',
            annotations = [
                dict(
                    x=0.225,
                    y=1,
                    xref='paper',
                    yref='paper',
                    text='Degree Distribution',
                    font = dict(
                        size = 16
                    ),
                    showarrow=False,
                    xanchor='center',
                    yanchor='bottom'
                ),
                dict(
                    x=0.775,
                    y=1,
                    xref='paper',
                    yref='paper',
                    text='Cumulative Degree Distribution',
                    font = dict(
                        size = 16
                    ),
                    showarrow=False,
                    xanchor='center',
                    yanchor='bottom'
                )
            ]
        )
        
        fig = go.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename= self.graph_name.capitalize())

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cmd = parse_args()
    Plot(cmd.dataset)
    #Plot_Distribution(DATASETS[cmd.dataset])


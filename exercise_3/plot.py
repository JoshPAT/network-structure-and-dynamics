#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Quan zhou'

import plotly.plotly as py
import plotly.graph_objs as go
from feature import Feature
from models import RandomFixedDegreeModels
import collections

class Plot(Feature):
    '''
    It is used to plot the properties of graph using plotly
    Some of codes here are not pythonic.
    '''

    def __init__(self, dataset):
        super(Plot, self).__init__(dataset)
        self.title = dataset.split('.')[0]

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
            title = self.title.capitalize(),
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
        plot_url = py.plot(fig, filename= self.title.capitalize())

class Plot_cc(RandomFixedDegreeModels):
    def __init__(self):
        super(Plot_cc, self).__init__(option = 'switch')
        self.plot_cc()

    def plot_cc(self):
        plotcc = {}
        with open('datasets/' + self.cluster_file, 'r') as f:
            for line in f.readlines():
                n, c = line.strip().split(' ')
                plotcc[int(n)] = float(c)
        plotcc = collections.OrderedDict(sorted(plotcc.items()))
        trace1 = go.Scatter(
            x = plotcc.keys(),
            y = plotcc.values(),
            showlegend = False,
            mode = 'lines+markers',
        )
        data = [trace1]
        layout = go.Layout(
            title = 'clustering coefficient Trends',
            autosize = True,
            
            xaxis = dict(
                type = 'log',
                autorange = True,
                title = 'Switch Times',
                exponentformat='power',
                tickangle = 10
            ),
            yaxis = dict(
                type = 'log',
                autorange = True,
                title = 'clustering coefficient',
                exponentformat ='power',
                tickangle = 10
            ),
        )
        fig = go.Figure(data = data, layout =layout)
        plot_url = py.plot(fig, filename= self.cluster_file.split('.')[0].capitalize())

def watts_strogaz():
    p1 = [x * 0.1 for x in xrange(10)]
    p2 = [x * 0.01 for x in xrange(10)]
    pc = p1 + p2 #combined probility
    pc = sorted(pc)
    ratio = {}
    c0 = measure_graph(models.model_WS(7235, 6, 0))
    for p in pc:
        ratio[p] = measure_graph(models.model_WS(7235, 6, p)) / c0
    ratio = collections.OrderedDict(sorted(ratio.items()))
    trace1 = go.Scatter(
            x = ratio.keys(),
            y = ratio.values(),
            name = 'C(p)/C(0)',
            showlegend = True,
            mode = 'lines+markers',
        )
    data = [trace1]
    layout = go.Layout(
        title = 'clustering coefficient Trends',
        autosize = True,
        xaxis = dict(
            type = 'log',
            autorange = True,
            title = 'Probability',
            exponentformat='none',
            tickangle = 10
        ),
        yaxis = dict(
            type = 'log',
            autorange = True,
            title = 'clustering coefficient Ratio',
            exponentformat ='none',
            tickangle = 10
        ),
        plot_bgcolor='rgb(238, 238, 238)',
    )
    fig = go.Figure(data = data, layout =layout)
    plot_url = py.plot(fig, filename= 'watts Strogaz Randomization Trends')


def measure_graph(m):
    if isinstance(m, str):
        p = Plot(m)
    else:
        p = Plot(m.file)
    p.compute_size()
    #p.plot_degree()
    return p.compute_triangle_values()

if __name__ == '__main__':
    import models, datasets
    watts_strogaz()
    #measure_graph(models.model_ER())
    #measure_graph(models.model_FD('switch'))
    #measure_graph('drosophila_PPI.txt')
    #measure_graph(models.model_BA())
    #measure_graph(models.model_WS(7235, 6, 0.1))
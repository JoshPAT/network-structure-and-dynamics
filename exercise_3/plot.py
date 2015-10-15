
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from nsd_homework_2 import Graph

class Plot(Graph):
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
            height = 500,
            
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

    def 

def measure_graph(m):
    graph = Plot(m.file)
    graph.compute_all()
    graph.plot_degree()
    return graph.compute_triangle_values()[1]

if __name__ == '__main__':
    import graphmodels, datasets
    #measure_graph(graphmodels.model_ER())
    measure_graph(graphmodels.model_FD())
    
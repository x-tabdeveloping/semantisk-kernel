from distutils.command.build import build
from typing import Dict, List

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from community import community_louvain
from gensim.models import Word2Vec
from networkx.drawing.layout import spring_layout

from semkern.kernel import build_kernel, distance_matrix


def get_edge_pos(edges: np.ndarray, x_y: np.ndarray) -> np.ndarray:
    """
    Through a series of nasty numpy tricks, that I® wrote
    this function transforms edges and either the x or the y positions of nodes to
    the x or y positions for the lines in the plotly figure.
    In order for the line not to be connected, the algorithm has to insert a nan value after each pair of points that have to be connected.
    """
    edges = np.array(edges)
    x_y = np.array(x_y)
    a = x_y[edges]
    a.shape
    b = np.zeros((a.shape[0], a.shape[1] + 1))
    b[:, :-1] = a
    b[:, -1] = np.nan
    return b.flatten()


def networkx_graph(kernel: List[str], distance_matrix: np.ndarray) -> Dict:
    """
    Returns a graph dict based on the established kernel and word distances.
    The output contains the following:
        - labels: all labels of the graph
        - edges: all edges of the graph
        - pos: positions of nodes
        - colors: the color of each node based on community partitioning
        - connections: Number of connections of each node, determines the size of the node on the graph
    """
    connections = np.sum(distance_matrix != 0, axis=1)
    distance_matrix = distance_matrix * 10  # scale
    dt = [("len", float)]
    distance_matrix = distance_matrix.view(dt)
    G = nx.from_numpy_matrix(distance_matrix)
    pos = spring_layout(nx.from_numpy_matrix(distance_matrix))
    parts = community_louvain.best_partition(G)
    colors = list(parts.values())
    edges = np.array(G.edges())
    return {
        "labels": kernel,
        "edges": edges,
        "pos": pos,
        "colors": colors,
        "connections": connections,
    }


def build_plot(graph: Dict) -> go.Figure:
    """
    Builds Plotly plot object based on the graph dictionary yielded by networkx_graph
    """
    x, y = zip(*graph["pos"].values())
    x, y = np.array(x), np.array(y)
    edges_x = get_edge_pos(graph["edges"], x)
    edges_y = get_edge_pos(graph["edges"], y)
    sum_connections = np.sum(graph["connections"])
    graph["connections"] = np.array(graph["connections"])
    indices = list(range(len(x)))
    size = 100 * graph["connections"] / sum_connections
    annotations = [
        dict(
            text=node,
            x=x[i],
            y=y[i],
            showarrow=False,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="rgba(0,0,0,0.5)",
            font={
                "family": "Helvetica",
                "size": max(size[i], 10),
                "color": "black",
            },
        )
        for i, node in enumerate(graph["labels"])
    ]
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        hoverinfo="text",
        text=graph["labels"],
        marker={
            "colorscale": "sunsetdark",
            "reversescale": True,
            "color": graph["colors"],
            "size": 10 * size,
            "line_width": 2,
        },
        customdata=indices,
    )
    edge_trace = go.Scatter(
        x=edges_x,
        y=edges_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            clickmode="event",
            annotations=annotations,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            titlefont_size=16,
            showlegend=False,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def plot(seeds: List[str], k: int, m: int, model: Word2Vec) -> go.Figure:
    """
    Creates and plots semantic kernel given the seeds, number of words to be yielded from the first and second level of association,
    and a precomputed word2vec model.
    """
    kernel = build_kernel(seeds, k, m, model)
    delta = distance_matrix(kernel, model)
    figure = build_plot(networkx_graph(kernel, delta))
    return figure

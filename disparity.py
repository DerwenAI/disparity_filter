#!/usr/bin/env python
# encoding: utf-8

from networkx.readwrite import json_graph
from os import path
from scipy.stats import percentileofscore
from util import report_error
import networkx as nx
import numpy as np
import pandas as pd
import sys


DEBUG = False # True

GRANULARITY = 100
MIN_DEGREE = 5 # 2


######################################################################
## disparity filter for extracting the multiscale backbone of
## complex weighted networks

def get_nes (graph, label):
    """
    find the neighborhood attention set (NES) for the given label
    """
    for node_id in graph.nodes():
        node = graph.node[node_id]

        if node["label"].lower() == label:
            return set([node_id]).union(set([id for id in graph.neighbors(node_id)]))


def disparity_integral (x, k):
    """
    calculate the definite integral for the PDF in the disparity filter
    """
    assert x != 1.0, "x == 1.0"
    assert k != 1.0, "k == 1.0"
    return ((1.0 - x)**k) / ((k - 1.0) * (x - 1.0))


def get_disparity_significance (norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    return 1.0 - ((degree - 1.0) * (disparity_integral(norm_weight, degree) - disparity_integral(0.0, degree)))


def disparity_filter (graph):
    """
    implements a disparity filter, based on multiscale backbone networks
    https://arxiv.org/pdf/0904.2389.pdf
    """
    disp_metrics = []
    
    for node_id in graph.nodes():
        node = graph.node[node_id]
        degree = node["degree"]
        strength = 0.0

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]
            strength += edge["weight"]

        node["strength"] = strength

        for id0, id1 in graph.edges(nbunch=[node_id]):
            edge = graph[id0][id1]

            norm_weight = edge["weight"] / strength
            edge["norm_weight"] = norm_weight

            if degree > 1:
                try:
                    if norm_weight == 1.0:
                        norm_weight -= 0.0001

                    disparity = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    report_error("disparity {}".format(repr(node)), fatal=True)

                edge["disparity"] = disparity
                disp_metrics.append(disparity)
            else:
                edge["disparity"] = 0.0

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]
        edge["disp_ptile"] = percentileofscore(disp_metrics, edge["disparity"]) / 100.0

    return disp_metrics


######################################################################
## metrics

def calc_quantiles (metrics, num=GRANULARITY):
    """
    calculate quantiles for the given list
    """
    bins = np.linspace(0, 1, num=num, endpoint=True)
    s = pd.Series(metrics)
    q = s.quantile(bins, interpolation="nearest")

    try:
        dig = np.digitize(metrics, q) - 1
    except ValueError as e:
        print("ValueError:", str(e), metrics, s, q, bins)
        sys.exit(-1)

    quantiles = []

    for idx, q_hi in q.iteritems():
        quantiles.append(q_hi)

        if DEBUG:
            print(idx, q_hi)

    return quantiles


def calc_centrality (graph):
    """
    to conserve compute costs, ignore centrality for nodes below MIN_DEGREE
    """
    global MIN_DEGREE

    sub_graph = graph.copy()
    sub_graph.remove_nodes_from([ n for n, d in graph.degree_iter() if d < MIN_DEGREE ])

    centrality = nx.betweenness_centrality(sub_graph, weight="weight")
    #centrality = nx.closeness_centrality(sub_graph, distance="distance")

    return centrality


if __name__ == "__main__":
    #graph_path = sys.argv[1]
    graph = nx.Graph()    

    # calculate the multiscale backbone metrics
    disp_metrics = disparity_filter(graph)

#!/usr/bin/env python
# encoding: utf-8

from networkx.readwrite import json_graph
from scipy.stats import percentileofscore
from traceback import format_exception
import cProfile
import json
import networkx as nx
import numpy as np
import pandas as pd
import pstats
import random
import sys

DEBUG = False # True


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
    alpha_measures = []
    
    for node_id in graph.nodes():
        node = graph.nodes[node_id]
        degree = graph.degree(node_id)
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

                    alpha = get_disparity_significance(norm_weight, degree)
                except AssertionError:
                    report_error("disparity {}".format(repr(node)), fatal=True)

                edge["alpha"] = alpha
                alpha_measures.append(alpha)
            else:
                edge["alpha"] = 0.0

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]
        edge["alpha_ptile"] = percentileofscore(alpha_measures, edge["alpha"]) / 100.0

    return alpha_measures


######################################################################
## related metrics

def calc_centrality (graph, min_degree=1):
    """
    to conserve compute costs, ignore centrality for nodes below `min_degree`
    """
    sub_graph = graph.copy()
    sub_graph.remove_nodes_from([ n for n, d in list(graph.degree) if d < min_degree ])

    centrality = nx.betweenness_centrality(sub_graph, weight="weight")
    #centrality = nx.closeness_centrality(sub_graph, distance="distance")

    return centrality


def calc_quantiles (metrics, num):
    """
    calculate `num` quantiles for the given list
    """
    global DEBUG

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


def calc_alpha_ptile (alpha_measures, show=True):
    """
    calculate the quantiles used to define a threshold alpha cutoff
    """
    quantiles = calc_quantiles(alpha_measures, num=10)
    num_quant = len(quantiles)

    if show:
        print("\tptile\talpha")

        for i in range(num_quant):
            percentile = i / float(num_quant)
            print("\t{:0.2f}\t{:0.4f}".format(percentile, quantiles[i]))

    return quantiles, num_quant


def cut_graph (graph, min_alpha_ptile=0.5, min_degree=2):
    """
    apply the disparity filter to cut the given graph
    """
    filtered_set = set([])

    for id0, id1 in graph.edges():
        edge = graph[id0][id1]

        if edge["alpha_ptile"] < min_alpha_ptile:
            filtered_set.add((id0, id1))

    for id0, id1 in filtered_set:
        graph.remove_edge(id0, id1)

    filtered_set = set([])

    for node_id in graph.nodes():
        node = graph.nodes[node_id]

        if graph.degree(node_id) < min_degree:
            filtered_set.add(node_id)

    for node_id in filtered_set:
        graph.remove_node(node_id)



######################################################################
## profiling utilities

def start_profiling ():
    """start profiling"""
    pr = cProfile.Profile()
    pr.enable()

    return pr


def stop_profiling (pr):
    """stop profiling and report"""
    pr.disable()

    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

    ps.print_stats()
    print(s.getvalue())


def report_error (cause_string, logger=None, fatal=False):
    """
    TODO: errors should go to logger, and not be fatal
    """
    etype, value, tb = sys.exc_info()
    error_str = "{} {}".format(cause_string, str(format_exception(etype, value, tb, 3)))

    if logger:
        logger.info(error_str)
    else:
        print(error_str)

    if fatal:
        sys.exit(-1)


######################################################################
## graph serialization

def load_graph (graph_path):
    """
    load a graph from JSON
    """
    with open(graph_path) as f:
        data = json.load(f)
        graph = json_graph.node_link_graph(data, directed=True)
        return graph


def save_graph (graph, graph_path):
    """
    save a graph as JSON
    """
    with open(graph_path, "w") as f:
        data = json_graph.node_link_data(graph)
        json.dump(data, f)


######################################################################
## testing

def random_graph (n, k, seed=0):
    """
    populate a random graph (with an optional seed) with `n` nodes and
    up to `k` edges for each node
    """
    graph = nx.DiGraph()
    random.seed(seed)

    for node_id in range(n):
        graph.add_node(node_id, label=str(node_id))

    for node_id in range(n):
        population = set(range(n)) - set([node_id])

        for neighbor in random.sample(population, random.randint(0, k)):
            weight = random.random()
            graph.add_edge(node_id, neighbor, weight=weight)

    return graph


def describe_graph (graph, min_degree=1, show_centrality=False):
    """
    describe a graph
    """
    print("\ngraph: {} nodes {} edges\n".format(len(graph.nodes()), len(graph.edges())))

    if show_centrality:
        print(calc_centrality(graph, min_degree))


def main (n=100, k=10, min_alpha_ptile=0.5, min_degree=2):
    # generate a random graph (from seed, always the same)
    graph = random_graph(n, k)

    save_graph(graph, "g.json")
    describe_graph(graph, min_degree)

    # calculate the multiscale backbone metrics
    alpha_measures = disparity_filter(graph)
    quantiles, num_quant = calc_alpha_ptile(alpha_measures)
    alpha_cutoff = quantiles[round(num_quant * min_alpha_ptile)]

    print("\nfilter: percentile {:0.2f}, min alpha {:0.4f}, min degree {}".format(
            min_alpha_ptile, alpha_cutoff, min_degree
            ))

    # apply the filter to cut the graph
    cut_graph(graph, min_alpha_ptile, min_degree)

    save_graph(graph, "h.json")
    describe_graph(graph, min_degree)


######################################################################
## main entry point

if __name__ == "__main__":
    main()

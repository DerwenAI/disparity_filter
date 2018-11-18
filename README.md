# disparity_filter

Implements a **disparity filter** in Python, using graphs in
[NetworkX](https://networkx.github.io/),
based on *multiscale backbone networks*:

"Extracting the multiscale backbone of complex weighted networks"  
M. Ángeles Serrano, Marián Boguña, Alessandro Vespignani  
https://arxiv.org/pdf/0904.2389.pdf

> The disparity filter exploits local heterogeneity and local correlations among weights to extract the network backbone by considering the relevant edges at all the scales present in the system. The methodology preserves an edge whenever its intensity is a statistically not compatible with respect to a null hypothesis of uniform randomness for at least one of the two nodes the edge is incident to, which ensures that small nodes in terms of strength are not neglected. As result, the disparity filter reduces the number of edges in the original network significantly keeping, at the same time, almost all the weight and a large fraction of nodes. As well, this filter preserves the cut-off of the degree distribution, the form of the weight distribution, and the clustering coefficient.

This project is similar to, albeit providing different features than:

  * https://github.com/aekpalakorn/python-backbone-network


## Implementation Details

If you are new to *multiscale backbone* analysis, think of this as
analogous to *centrality* calculated on the edges of a graph rather
than its nodes. In other words, as a "dual" of the problem typically
faced in social networks. By managing cuts through a process of
iterating between measures of *centrality* and *disparity*
respectively, one can scale a large, noisy graph into something more
amenable for work with ontology.

The code expects each *node* to have a required *label* attribute,
which is a string unique within all of the nodes in the graph. Each
*edge* is expected to have a *weight* attribute, a decimal in the
range of `[0.0, 1.0]` which represents the relative weight of that
edge's relationship.

After calculating the disparity metrics, each node get assigned a
*strength* attribute, which is the sum of its edges' weights. Each
edge gets assigned the following attributes:

  * *norm_weight*: ratio of the `edge[weight] / node[strength]`
  * *alpha*: disparity *alpha* metric
  * *alpha_ptile*: percentile for *alpha*, compared across the graph

One important distinction is that this implementation comes from work
in NLP and ontology, where graphs tend to become relatively "noisy"
and there are many graphs generated through automation which need to
be filtered. NLP applications had tended to reuse graph techniques
from social graph analysis, such as *connected components*,
*centrality*, cuts based on the relative *degree* of nodes -- while
applications which combine NLP plus ontology tend to need information
based on the edges.

In particular, this implementation focuses on directed graphs, and
uses quantile analysis to adjust graph cuts. The original paper showed
how to make cuts using the raw *alpha* values, which depended on
manual (human) decisions.  However, that is less than ideal for
applications in machine learning, where more automation is typically
required. Use of quantiles allows for a form of "normalization" for
threshold values, so that cuts can be performed more consistently when
automated.

This implementation also integrates support for working with
*neighborhood attention sets* (NES) and other mechanisms for working
with semantics and ontologies.


## Getting Started

```
pip install -r requirements.txt
```

## Example

The running default `main()` function:
```
python disparity.py
```

That will:

  1. generate a random graph (using a seed) of 100 nodes, each with < 10 edges
  2. calculate the significance (*alpha*) for the disparity filter
  3. calculate quantiles for *alpha*
  4. cut edges below the 50th percentile (median) for *alpha*
  5. cut nodes with degree < 2

```
graph: 100 nodes 489 edges

   ptile     alpha
   0.00	     0.0000
   0.10	     0.0305
   0.20	     0.0624
   0.30	     0.1027
   0.40	     0.1512
   0.50	     0.2159
   0.60	     0.3222
   0.70	     0.4821
   0.80	     0.7102
   0.90	     0.9998

filter: percentile 0.50, min alpha 0.2159, min degree 2

graph: 89 nodes 235 edges
```

In practice, adjust those thresholds as needed before making a cut on
a graph. This mechanism provides a "dial" to adjust the scale of the
multiscale backbone of the graph.


## Contributors

Please use the `Issues` section to ask questions or report any problems.
Before submitting a *pull request* be sure to sign the
[CLA](http://contributoragreements.org/u2s/222mlog137).

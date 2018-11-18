# disparity_filter

Implements a **disparity filter** in Python, using graphs in
[NetworkX](https://networkx.github.io/),
based on *multiscale backbone networks*:

"Extracting the multiscale backbone of complex weighted networks"  
M. Ángeles Serrano, Marián Boguña, Alessandro Vespignani  
https://arxiv.org/pdf/0904.2389.pdf

> The disparity filter exploits local heterogeneity and local correlations among weights to extract the network backbone by considering the relevant edges at all the scales present in the system. The methodology preserves an edge whenever its intensity is a statistically not compatible with respect to a null hypothesis of uniform randomness for at least one of the two nodes the edge is incident to, which ensures that small nodes in terms of strength are not neglected. As result, the disparity filter reduces the number of edges in the original network significantly keeping, at the same time, almost all the weight and a large fraction of nodes. As well, this filter preserves the cut-off of the degree distribution, the form of the weight distribution, and the clustering coefficient.

Think of this as "centrality" calculated on the edges of a graph
rather than its nodes.

This project is similar to, albeit providing different features than:

  * https://github.com/aekpalakorn/python-backbone-network


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
G: 100 nodes 489 edges

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

G: 89 nodes 235 edges
```

In practice, adjust those thresholds as needed before making a cut on a graph.
This mechanism provides a "dial" to scale the multiscale backbone of the graph.

## Contributors

Please use the `Issues` section to ask questions or report any problems.
Before submitting a *pull request* be sure to sign the
[CLA](http://contributoragreements.org/u2s/222mlog137).

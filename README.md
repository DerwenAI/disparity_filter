# disparity_filter

Implements a **disparity filter** in Python, using graphs in NetworkX,
based on *multiscale backbone networks*:

"Extracting the multiscale backbone of complex weighted networks"  
M. Ángeles Serrano, Marián Boguña, Alessandro Vespignani  
https://arxiv.org/pdf/0904.2389.pdf

Think of this as if it were "centrality" calculated on the edges of a
graph rather than its nodes.

Similar to:

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
  2. calculate the significance (alpha) for the disparity filter
  3. calculate quantiles for alpha
  4. cut edges below the 80th percentile for alpha
  5. cut nodes with degree < 2

```
G: 100 nodes 489 edges
filter:	min alpha 0.5472, min degree 2
G: 86 nodes 210 edges
```

In practice, adjust those thresholds as needed before making a cut on a graph.
This mechanism provides a "dial" to scale the multiscale backbone of the graph.

## Contributors

Please use the `Issues` section to ask questions or report problems, and be sure to sign the [CLA](http://contributoragreements.org/u2s/222mlog137) before submitting a PR.

# disparity_filter

Implements a **disparity filter** in Python, using graphs in NetworkX,
based on *multiscale backbone networks*:

"Extracting the multiscale backbone of complex weighted networks"  
M. Ángeles Serrano, Marián Boguña, Alessandro Vespignani  
https://arxiv.org/pdf/0904.2389.pdf

Think of this as if it were "centrality" calculated on the edges of a
graph rather than its nodes.


## Getting Started

```
pip install -r requirements.txt
```

## Example Use

The running default `main()` function:
```
python disparity.py
```

Will generate a random graph (using a seed) of 100 nodes, then prune
all edges below the 80th percentile for the disparity measure and
prune all nodes with degree < 2:
```
G: 100 nodes 489 edges
filter:	      min disparity cutoff 0.5472, min degree 2
G: 86 nodes 210 edges
```

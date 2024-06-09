Base Models
=============

The DGRL-Hardware toolbox currently supports  GNN backbones/graph transformers, to call each method the corresponding entry in general configuration one may set:

+--------------------------+----------------+----------------+
| GNN backbone/transformer | mesage passing | name in config |
+==========================+================+================+
| DGCN                     |     \-         | DGCN           |
|                          |                |                |
| DiGCN                    |     \-         | DiiGCN         |
|                          |                |                |
| MagNet                   |     \-         | MSGNN          |
|                          |                |                |
| GCN                      | undirected     | GCN            |
|                          |                |                |
| GCN                      | directed       | DIGCN          |
|                          |                |                |
| GCN                      | bidirected     | BIGCN          |
|                          |                |                |
| GIN(E)                   | undirected     | GIN(E)         |
|                          |                |                |
| GIN(E)                   | directed       | DIGIN(E)       |
|                          |                |                |
| GIN(E)                   | bidirected     | BIGIN(E)       |
|                          |                |                |
| GAT                      | undirected     | GAT            |
|                          |                |                |
| GAT                      | directed       | DIGAT          |
|                          |                |                |
| GAT                      | bidirected     | BIGAT          |
|                          |                |                |
| GPS-T                    | undirected     | GPS            |
|                          |                |                |
| GPS-T                    | directed       | DIGPS          |
|                          |                |                |
| GPS-T                    | bidirected     | BIGPS          |
|                          |                |                |
| GPS-P                    | undirected     | PERFORMER      |
|                          |                |                |
| GPS-P                    | directed       | DIPERFORMER    |
|                          |                |                |
| GPS-P                    | bidirected     | BIPERFORMER    |
+--------------------------+----------------+----------------+

The implementation of each methods are in ./models/base_model.py <>`_.



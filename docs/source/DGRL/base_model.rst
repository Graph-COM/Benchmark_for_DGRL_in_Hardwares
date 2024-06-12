Base Models
=============

The interface of each base model:

GNN Backbones/ Graph Transformers
-------------------------------------

DGCN
~~~~~~

.. code-block:: python
      def 
.. automodule:: DGRL-Hardware.models.base_model
    :members:
    :exclude-members:

The method is from `Directed Graph Convolutional Network <https://arxiv.org/abs/2004.13970>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.

DiGCN 
~~~~~~

The method is from `Digraph Inception Convolutional Networks <https://proceedings.neurips.cc/paper/2020/hash/cffb6e2288a630c2a787a64ccc67097c-Abstract.html>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.

Magnet
~~~~~~

The method is from `MagNet: A Neural Network for Directed Graphs <https://arxiv.org/abs/2102.11391>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.

GCN
~~~~~~

The method is from `Semi-Supervised Classification with Graph Convolutional Networks  <https://arxiv.org/abs/1609.02907>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GIN(E)
~~~~~~

The method is from `How Powerful are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GAT
~~~~~~

The method is from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GPS-T
~~~~~~

The method is from `Recipe for a General, Powerful, Scalable Graph Transformer <https://arxiv.org/abs/2205.12454>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GPS-P
~~~~~~

The method is from `Rethinking Attention with Performers <https://arxiv.org/abs/2009.14794>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

Message Passing Directions
------------------------------

- undirected (-)

- directed (DI-)

- bidirected (BI-)


The detailed implementation of each methods are in `./models/base_model.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/models/base_model.py>`_.



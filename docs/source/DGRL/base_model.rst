Base Models
=============

The interface of each base model:

GNN Backbones/ Graph Transformers
-------------------------------------

DGCN (DGCN in config)
~~~~~~



.. code-block:: python

      class DGCNConv(improved: bool = False, cached: bool = False, add_self_loops: bool = True, normalize: bool = True, **kwargs)

            def forward(x: torch.Tensor, edge_index: Union[torch.Tensor, torch_sparse.tensor.SparseTensor], edge_weight: Optional[torch.Tensor] = None) → torch.Tensor

The method is from `Directed Graph Convolutional Network <https://arxiv.org/abs/2004.13970>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.

DiGCN (DiiGCN in config)
~~~~~~

.. code-block:: python

      class DiGCNConv(in_channels: int, out_channels: int, improved: bool = False, cached: bool = True, bias: bool = True, **kwargs)
            
            def forward(x: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None) → torch.FloatTensor

The method is from `Digraph Inception Convolutional Networks <https://proceedings.neurips.cc/paper/2020/hash/cffb6e2288a630c2a787a64ccc67097c-Abstract.html>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.

Magnet (MSGNN in config)
~~~~~~
.. code-block:: python

      class MSConv(in_channels: int, out_channels: int, K: int, q: float, trainable_q: bool, normalization: str = 'sym', bias: bool = True, cached: bool = False, absolute_degree: bool = True, **kwargs)
            
            def forward(x_real: torch.FloatTensor, x_imag: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: Optional[torch.Tensor] = None, lambda_max: Optional[torch.Tensor] = None) → torch.FloatTensor

The method is from `MagNet: A Neural Network for Directed Graphs <https://arxiv.org/abs/2102.11391>`_. The implementation adopts the `PyGSD library <https://pytorch-geometric-signed-directed.readthedocs.io>`_.


GCN (GCN in config)
~~~~~~

.. code-block:: python

      class GCNConv(in_channels: int, out_channels: int, improved: bool = False, cached: bool = False, add_self_loops: Optional[bool] = None, normalize: bool = True, bias: bool = True, **kwargs)

            def forward(x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None)→ Tensor

The method is from `Semi-Supervised Classification with Graph Convolutional Networks  <https://arxiv.org/abs/1609.02907>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GIN(E) (GIN, GINE in config)
~~~~~~

.. code-block:: python

      class GINConv(nn: Callable, eps: float = 0.0, train_eps: bool = False, **kwargs)

            def forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], edge_index: Union[Tensor, SparseTensor], size: Optional[Tuple[int, int]] = None)→ Tensor

      class GINEConv(nn: Module, eps: float = 0.0, train_eps: bool = False, edge_dim: Optional[int] = None, **kwargs)

            def forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], edge_index: Union[Tensor, SparseTensor], edge_attr: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None)→ Tensor

The method is from `How Powerful are Graph Neural Networks? <https://arxiv.org/abs/1810.00826>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GAT (GAT in config)
~~~~~~

.. code-block:: python

      class GATConv(in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True, edge_dim: Optional[int] = None, fill_value: Union[float, Tensor, str] = 'mean', bias: bool = True, **kwargs)

            def forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], edge_index: Union[Tensor, SparseTensor], edge_attr: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None, return_attention_weights: Optional[Tensor] = None)→ Tensor


The method is from `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GPS-T (GPS in config)
~~~~~~

.. code-block:: python

      class GPSConv(channels: int, conv: Optional[MessagePassing], heads: int = 1, dropout: float = 0.0, act: str = 'relu', act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[str] = 'batch_norm', norm_kwargs: Optional[Dict[str, Any]] = None, attn_type: str = 'multihead', attn_kwargs: Optional[Dict[str, Any]] = None)

            def forward(x: Tensor, edge_index: Union[Tensor, SparseTensor], batch: Optional[Tensor] = None, **kwargs)→ Tensor

The method is from `Recipe for a General, Powerful, Scalable Graph Transformer <https://arxiv.org/abs/2205.12454>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

GPS-P (PERFORMER in config)
~~~~~~

.. code-block:: python

      class GPSConv(channels: int, conv: Optional[MessagePassing], heads: int = 1, dropout: float = 0.0, act: str = 'relu', act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[str] = 'batch_norm', norm_kwargs: Optional[Dict[str, Any]] = None, attn_type: str = 'performer', attn_kwargs: Optional[Dict[str, Any]] = None)

            def forward(x: Tensor, edge_index: Union[Tensor, SparseTensor], batch: Optional[Tensor] = None, **kwargs)→ Tensor


The method is from `Rethinking Attention with Performers <https://arxiv.org/abs/2009.14794>`_. The implementation adopts the `PyG library <https://pytorch-geometric.readthedocs.io>`_.

Message Passing Directions
------------------------------

- undirected (-)

for undirected message passing, set directed=0 in the general config and implement the forward function with undirected message passing:

.. code-block:: yaml

      #general config.yaml
      train:
            directed: 0

.. code-block:: python

      def __init__():
            self.conv = $model
      def forward():
            x = self.conv(x, edge_index)
      

- directed (DI-)

.. code-block:: yaml

      #general config.yaml
      train:
            directed: 1

.. code-block:: python

      def __init__():
            self.conv = $model
      def forward():
            x = self.conv(x, edge_index)

- bidirected (BI-)

.. code-block:: yaml

      #general config.yaml
      train:
            directed: 1

.. code-block:: python

      def __init__():
            self.forward_conv = $model
            self.backward_conv = $model
      def forward():
            x1 = self.forward_conv(x, edge_index)
            x2 = self.backward_conv(x, edge_index)
            x = merge(x1 + x2)


The detailed implementation of each methods are in `./models/base_model.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/models/base_model.py>`_.



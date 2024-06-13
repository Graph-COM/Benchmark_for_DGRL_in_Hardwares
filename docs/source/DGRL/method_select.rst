Select from an Existing Method from Configuration
===================================================



The DGRL-Hardware toolbox provides the implementation of 21 existing combinations of backbone/message passing direction and positional encoding (PE) incorporation. One can call one of these methods by editing the general configuration (listed in `./configs/general/ <https://github.com/Graph-COM/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/configs/general>`_) and the PE configuration (listed in `./configs/pe/ <https://github.com/Graph-COM/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/configs/pe>`_, 

An example of the genral config can be as follows:

.. code-block :: yaml

    # general_config.yaml      
      model: 
        name: BIGAT
        hidden_dim: 192
        num_layers: 3
        node_input_dim: 7
        edge_input_dim: 2
        dropout: 0.2
        mlp_out:
          num_layer: 4


The name of all the methods to call are listed in the table below, for a detailed introduction on the interface and implementation of each base model, please refer to `Introduction on base DGRL methods <base_model.html>`_:


+--------------------------+----------------+----------------+
| GNN backbone/transformer | mesage passing | name in config |
+==========================+================+================+
| DGCN                     | directed       | DGCN           |
|                          |                |                |
| DiGCN                    | directed       | DiiGCN         |
|                          |                |                |
| MagNet                   | directed       | MSGNN          |
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

An example to config PE is shown as follows:


.. code-block :: yaml
    
    # pe_config.yaml
      model:
        pe_file_name: maglap_1q_spe
        pe_type: maglap
        q: 0.1
        q_dim: 1
        pe_strategy: invariant_fixed
        pe_encoder: spe
        mag_pe_dim_input: 10
        mag_pe_dim_output: 10
        se_pe_dim_input: 0
        se_pe_dim_output: 0
        
        eigval_encoder:
          in: 1
          hidden: 32
          out: 8
          num_layer: 3

The table below show the configuration to use for magnetic Laplacian PE with NPE or EPE:

+--------+-------------+---------+-----------------+-------------+---------------------------------------+
| stable | potential q | pe_type | pe_strategy     | pe_embedder | example                               |
+========+=============+=========+=================+=============+=======================================+
| NPE    | q=0         | lap     | variant         | naive       | ./configs/pe/lap10/lap_naive          |
|        |             |         |                 |             |                                       |
| NPE    | q>0         | maglap  | variant         | naive       | ./configs/pe/maglap10/maglap_1q_naive |
|        |             |         |                 |             |                                       |
| EPE    | q=0         | lap     | invariant_fixed |             | ./configs/pe/lap10/lap_spe            |
|        |             |         |                 |             |                                       |
| EPE    | q>0         | maglap  | invariant_fixed |             | ./configs/pe/maglap10/maglap_1q_spe   |
+--------+-------------+---------+-----------------+-------------+---------------------------------------+


The eigval_encoder is used to configure the hyper-parameters of stable PE.

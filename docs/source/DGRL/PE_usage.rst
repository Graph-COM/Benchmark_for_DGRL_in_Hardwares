PE usage
=========

We provide two ways of incorporating PEs, node PE (NPE) and edge PE (EPE), by simply adding a configuration of PE. 

The configurations can be found in `./configs/pe/ <https://github.com/Graph-COM/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL_Hardware/configs/pe>`_, and the implementations can be found in `./models/middle_model.py <https://github.com/Graph-COM/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL_Hardware/models/middle_model.py>`_.


A sample config can be like:

.. code-block:: yaml
  
  model:
  pe_file_name: lap_naive
  pe_type: lap
  pe_strategy: variant
  lap_pe_dim_input: 10
  lap_pe_dim_output: 10
  se_pe_dim_input: 0
  se_pe_dim_output: 0
  
  eigval_encoder:
    in: 1
    hidden: 32
    out: 8
    num_layer: 3

  pe_embedder:
    name: naive
    

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

Note that NPE directly concatenate PE with node feature, while EPE processes PE with `stable PE <https://arxiv.org/abs/2310.02579>`_ and concatenates PE on edge features.

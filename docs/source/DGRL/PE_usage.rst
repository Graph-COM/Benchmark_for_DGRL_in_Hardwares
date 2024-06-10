PE usage
=========

We provide two ways of incorporating PEs, node PE (NPE) and edge PE (EPE), by simply adding a configuration of PE. 

The configurations can be found in `./configs/pe/ <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/configs/pe>`_, and the implementations can be found in `./models/middle_model.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/models/middle_model.py>`_.


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
    

Note that NPE directly concatenate PE with node feature, while EPE processes PE with `stable PE <https://arxiv.org/abs/2310.02579>`_ and concatenates PE on edge features.

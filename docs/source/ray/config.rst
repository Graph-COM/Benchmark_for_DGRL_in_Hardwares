Hyper-Parameter Search Space
==============================


Users could design the hyper-parameter search space simply with a config, for the config we used for our benchmark, please refer to `./configs/ray/ <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/configs/ray>`_.

A sample config of RAY to search the hyper-parameter space is as follows:

.. code-block:: yaml

    name: BIGINE
    hidden_dim: [2,7]
    num_layers: [3,8]
    # for HLS lr: [1e-4, 5e-3]
    lr: [1e-4, 1e-2]
    batch_size: [64, 128, 256, 512, 1024]
    dropout: [0, 0.1, 0.2, 0.3]
    # for HLS node_input_dim: 7
    pe_dim_input: 20
    pe_dim_output: 8
    
    mlp_out:
      num_layer: [2,5]

Cnnfiguration for Models (General Configuration)
=================================================

Users can config a method involved in our tollbox. 

All the configurations searched with RAY for the datasets are provided in `./configs/general <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/configs/general>`_.
An example of the configuration file is as follows:

.. code-block:: yaml

      # sample_config.yaml

      utils:
        seed: 0
        torch_num_threads: 10
      
      task:
        name: 'AMP'
        type: AMP # type: cdfg or dfg
        target: bw # different target prediction
        processed_folder: '~/DGRL-Hardware/data_processed/'
        divide_seed: default #set as default or a seed
        raw_data_path: '~/DGRL-Hardware/data_raw/AMP/'
        data_processor: AMPDataProcessor
      
      train:
        batch_size: 1024
        epoch: 300
        optimizer: Adam
        scheduler: 
          name: StepLR
          step_size: 100
          gamma: 0.95
        lr : 0.007535530155915536
        criterion: MSE
        wandb: 0
        train_files: '~/DGRL-Hardware/train_files/'
        directed: 1
      
      model: 
        name: BIGAT
        hidden_dim: 192
        num_layers: 3
        node_input_dim: 7
        edge_input_dim: 2
        dropout: 0.2
        mlp_out:
          num_layer: 4

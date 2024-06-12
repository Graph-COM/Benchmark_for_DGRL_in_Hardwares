Tune with RAY after Model Configuration
========================================


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



An example command to tune with ray could be:

.. code-block:: sh
    
    nohup python -u main.py --mode tune --general_config amp/gain/bigine --pe_config lap10/lap_spe \
    --ray_config BIGINE --device '1,3,4,5' --num_gpu_per_trial 0.24 \
    >./ray_amp_gain_bigine_lap10_lap_spe.log 2>&1 </dev/null &


The scripts to tune the methods from our benchmark is provided in `./tune.sh <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/tune.sh>`_.


In each Runner, we use similar functions to call RAY for hyper-parameter search, a sample of the function is as follows:

.. code-block:: python

    def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
        reporter = CLIReporter(parameter_columns=['hidden_dim'],metric_columns=['loss', 'mse', 'r2'])
        # init ray tune
        dropout_p = hp.choice('dropout_p', tune_config['dropout'])
        if self.config['model'].get('pe_file_name') in ['lap_naive', 'maglap_1q_naive'] and self.config['model']['name'] in ['GPS', 'GPSSE', 'DIGPS', 'BIGPS']:
            hidden_dim = 14 + 28 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        else: 
            hidden_dim = 28 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        tune_parameter_config = {
        'name': tune_config['name'],
        'batch_size': hp.choice('batch_size', tune_config['batch_size']),
        'hidden_dim': hidden_dim,
        'num_layers': hp.randint('num_layers', int(tune_config['num_layers'][0]), int(tune_config['num_layers'][1])),
        'lr': hp.uniform('lr', float(tune_config['lr'][0]), float(tune_config['lr'][1])),
        'dropout': dropout_p,
        'mlp_out': {'num_layer': hp.randint('mlp_out', int(tune_config['mlp_out']['num_layer'][0]), 
                                            int(tune_config['mlp_out']['num_layer'][1]))},
        'node_input_dim': self.config['model']['node_input_dim'],
        'edge_input_dim': self.config['model']['edge_input_dim'],
        'pe_dim_input': tune_config['pe_dim_input'],
        'pe_dim_output': tune_config['pe_dim_output'],
        'criterion': 'MSE',
        'attn_type': 'multihead',
        'attn_kwargs': {'dropout': dropout_p},
        }
        tune_parameter_config = {**self.config['model'], **tune_parameter_config}
        scheduler = ASHAScheduler(
            max_t=800,
            grace_period=80,
            reduction_factor=2)
        
        hyperopt_search = HyperOptSearch(tune_parameter_config, metric='mse', mode='min')
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_ray),
                resources={'cpu': num_cpu, 'gpu': num_gpu_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='mse',
                mode='min',
                scheduler=scheduler,
                num_samples=num_samples,
                search_alg=hyperopt_search,   
            ),
            run_config=RunConfig(progress_reporter=reporter),
        )
        results = tuner.fit()
        
        best_result = results.get_best_result('mse', 'min')


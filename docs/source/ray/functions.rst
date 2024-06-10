The function to call RAY
=========================


In each runner, we use similar functions to call RAY for hyper-parameter search, a sample of the function is as follows:

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

Select from an existing dataset
=================================


To select from one of the existing `datasets and tasks <intro.html>`_, one may configure the selected dataset in general configs, an example could be:

.. code-block:: yaml
    
    task:
      name: 'HLS'
      type: cdfg # type: cdfg or dfg
      target: dsp # different target prediction
      processed_folder: '~/DGRL-Hardware/data_processed/'
      divide_seed: default #set as default or a seed
      raw_data_path: '~/DGRL-Hardware/data_raw/HLS/'
      data_processor: HLSDataProcessor

here the name gives the name of the dataset, type and target determines the task, the processed_folder defines the path to save the processed PyG format data, raw_data_path provides the path of the original data, the data_processor defines the name of the data processor to process the data.

The data processor for the existing datasets are implemented in `./data_processor/ <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/data_processor>`_.
For more details on how to customize the dataset including the data processor, please refer to `customize new datasets <data_customize.html>`_.

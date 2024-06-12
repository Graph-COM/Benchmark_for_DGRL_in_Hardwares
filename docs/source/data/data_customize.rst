Customize new datasets and tasks
===================================

To handle the new datasets, one needs to 1) process the data into either standard raw-data format or the PyG compatible format, and 2) customize a dataprocessor 3) customize a runner for the dataset.



The Data Format that DGRL-Hardware asscpets
----------------------------------------------


Raw Data
~~~~~~~~~~~

DGRL accepts the following csv format to store the raw data:

+-------------------+-----------------------------------------+
| file name         | description                             |
+===================+=========================================+
| edge.csv          | saves all the edges                     |
|                   |                                         |
| node-feat.csv     | saves all the edge feature              |
|                   |                                         |
| num-edge-list.csv | saves the number of edges in each graph |
|                   |                                         |
| num-node-list.csv | saves the number of nodes in each graph |
|                   |                                         |
| edge-feat.csv     | saves the edge features of each graph   |
|                   |                                         |
| flexible          | may save the labels                     |
+-------------------+-----------------------------------------+


Examples of the raw data can be found at `./data_raw/ <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/data_raw>`_.


DataProcessor to handle the raw data
--------------------------------------

One need to customize a Data processor to process the raw data into PyG compatible data. The file name should be NEWDATA_data_processor.py (e.g. AMP_data_processor), saved in the folder `./data_processor <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/data_processor>`_.

A tutorial to customize such data processor is as follows:

.. code-block:: python

    class [$NEWDATA]DataProcessor(InMemoryDataset):
      def __init__(self, config, mode):
          # one may directly follow/copy the implementation of the initialization of data processors in existing datasets
      def process(self):
          # here to process the raw data into the PyG compatible data format
          
      def read_csv_graph_raw(self, raw_dir, check_repeat_edge):
          # this is the key function to process .csv files into the PyG data, 
          # for setails please see https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/data_processor.

Runner to Run with the New Datasets
-------------------------------------

One also needs to customize a Runner to run with the new dataset. The file name should be NEWDATA_runner.py (e.g. AMP_runner.py). saved in the folder `./runner <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/tree/main/DGRL-Hardware/runner>`_.

A tutorial to customize such runner is as follows:


.. code-block:: python

    class [$NewData]Runner():
      def __init__(self, config):
          # one may follow/copy the implementation of any existing DatasetRunners.
      def train_ray(self, tune_parameter_config):
          # the function that tuning with RAY would call for training, one may refer to the implementation of existing datasets
          # All the datasets/tasks share almost the same implementation
      
      def train(self):
          # the function that evaluation would call for training, one may refer to the implementation of existing datasets
          # All the datasets/tasks share almost the same implementation

      def raytune(self):
          # the function to load hyper-parameter design space
         # All the datasets/tasks share almost the same implementation
  
      def train_one_epoch(self, data_loader, mode, epoch_idx):
          # One may need to customize due to the difference in evaluation metrics
       
      def test(self, load_statedict = True, test_num_idx = 0):
          # One may need to customize due to the difference in evaluation metrics

      

The other functions in the Runner class share the same implementation across the datasets.
    


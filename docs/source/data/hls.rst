High-Level Synthesis (HLS)
============================

Overview
-----------

HLS is originally from `High-Level Synthesis Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing <https://arxiv.org/abs/2201.06848>`_.

After HLS front-end compilation, six node features are extracted, as summarized in the table below:

+------------------+-------------------------------------------------+--------------------------------------+
| Feature          | Description                                     | Values                               |
+==================+=================================================+======================================+
| Node type        | General node type                               | operation nodes, blocks, ports, misc |
|                  |                                                 |                                      |
| Bitwidth         | Bitwidth of the node                            | 0-256, misc                          |
|                  |                                                 |                                      |
| Opcode type      | Opcode categories based on LLVM                 | binary\_unary, bitwise, memory, etc. |
|                  |                                                 |                                      |
| Opcode           | Opcode of the node                              | load, add, xor, icmp, etc.           |
|                  |                                                 |                                      |
| Is start of path | Whether the node is the starting node of a path | 0, 1, misc                           |
|                  |                                                 |                                      |
| Cluster group    | Cluster number of the node                      | -1 - 256, misc                       |
+------------------+-------------------------------------------------+--------------------------------------+

Each edge has two features, the edge type represented in integers, and a binary value indicating whether this edge is a back edge.
Each graph is labeled based on its post-implementation performance metrics, which are synthesized by `Vitis HLS <https://www.google.com/search?client=safari&rls=en&q=vitis+hls&ie=UTF-8&oe=UTF-8>`_ and implemented by `Vivado <https://www.xilinx.com/products/design-tools/vivado.html>`_.
Three metrics are used for regression: DSP, LUT, and CP.
The first two are integer numbers indicating the number of resources used in the final implementation; the last one is CP timing in fractional number, determining the maximum working frequency of FPGA.
The DFG and CDFG datasets consists of 19,120 and 18,570 C programs, respectively.
The figure below shows an example C program from the CDFG dataset, with the corresponding control dataflow graph shown in the right.
More information can be found in the original paper.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: hls/fig/HLS_cdfg.png
     - .. image:: hls/fig/hls_example_program.png



Interface
------------

Runner
~~~~~~~~

.. code-block:: python
  
    class HLSRunner():
        def __init__(self, config):
          # init takes a config
        def train_ray(self, tune_parameter_config):
          # function to implement training when tuning with ray
        def train(self):
          # function to implement training when evaluation
        def train_one_epoch(self, data_loader, mode, epoch_idx):
          # function that do back propogation for one epoch
        def test(self, load_statedict = True, test_num_idx = 0):
          # function for testing
        def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
          # main function to take the hyper-parameter search space in RAY
        
        

Details are in `./runner/HLS_runner.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/runner/HLS_runner.py>`_.

DataProcessor
~~~~~~~~~~~~~~~

.. code-block:: python
  
    class HLSDataProcessor(InMemoryDataset):
        def __init__(self, config, mode):
          # init takes a config, mode takes from `tune' for tuning, `get_result' for evaluation
        def process(self):
          # key functions to implement HLS data processing
        def read_csv_graph_raw(self, raw_dir, check_repeat_edge):
          # key function to process raw data into PyG data

Details are in `./data_processor/HLS_data_processor.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/data_processor/HLS_data_processor.py>`_.

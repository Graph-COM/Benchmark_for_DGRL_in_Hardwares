Datasets and Tasks Overview
===========================

The DGRL-Hardware benchmark currently select 5 datasets with 13 tasks in total fronm the hardware design loop, as illustrate in the figure below:

.. image:: fig/line.pdf
   :alt: Selected Datasets and Tasks


High-Level Synthesis (HLS)
___

HLS is originally from `High-Level Synthesis Performance Prediction using GNNs: Benchmarking, Modeling, and Advancing <https://arxiv.org/abs/2201.06848>`_, 

After HLS front-end compilation, six node features are extracted, as summarized below.
Each edge has two features, the edge type represented in integers, and a binary value indicating whether this edge is a back edge.
Each graph is labeled based on its post-implementation performance metrics, which are synthesized by Vitis HLS and implemented by Vivado.
Three metrics are used for regression: DSP, LUT, and CP.
The first two are integer numbers indicating the number of resources used in the final implementation; the last one is CP timing in fractional number, determining the maximum working frequency of FPGA.
The DFG and CDFG datasets consists of 19,120 and 18,570 C programs, respectively.
The figure below shows an example C program from the CDFG dataset, with the corresponding control dataflow graph shown in the right.
More information can be found in the original paper.

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. image:: hls/fig/HLS_cdfg.pdf
         :width: 200px
     - .. image:: hls/fig/hls_example_program.pdf
         :width: 200px


  




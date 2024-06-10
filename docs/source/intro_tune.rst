Tune with RAY after Model Configuration
========================================


.. figure:: fig/code_frame.png

An example command totune with ray could be:

.. code-block:: sh
    
    nohup python -u main.py --mode tune --general_config amp/gain/bigine --pe_config lap10/lap_spe \
    --ray_config BIGINE --device '1,3,4,5' --num_gpu_per_trial 0.24 \
    >./ray_amp_gain_bigine_lap10_lap_spe.log 2>&1 </dev/null &


The scripts to tune the methods from our benchmark is provided in `./tune.sh <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/tune.sh>`_.

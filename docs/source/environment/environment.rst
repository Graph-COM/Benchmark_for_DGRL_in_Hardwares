
Environment
=====

.. _installation:

Installation
------------
We recommend environment management with [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html). The supports below are based on Conda.

To config a DGRL method and train the model on existing datasets, the following packages are required:

.. code-block:: shell

    conda create -n dgrl python==3.10
    conda activate dgrl
    # torch
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    # PyG (>=2.4.0)
    pip install torch_geometric==2.4.0
    # PyG dependencies
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
    # PyGSD
    pip install torch-geometric-signed-directed
    # RAY
    pip install -U "ray[data,train,tune,serve]"
    # RAY dependencies
    pip install hyperopt
    # other dependencies
    pip install prettytable
    pip install torchmetrics
    pip install hyperopt
    pip install easydict
    
    
    
    
    

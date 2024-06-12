To customize a new method
===========================

To customize a new methods other than existing base models, one should give the name and implementation in `./models/base_model.py <https://github.com/peterwang66/Benchmark_for_DGRL_in_Hardwares/blob/main/DGRL-Hardware/models/base_model.py>`_, which controls the use of backbone models.


Specifically, one gives the inititalization: 

.. code-block:: python

    class BaseModel(torch.nn.Module):
        def __init__(self, **kwargs):
            if self.base_model == $New_method:
            # define the init of new method
                self.conv = #New_conv

            # an example could be:
            if self.base_model in ['GINE', 'DIGINE']:
                nn = Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    BatchNorm1d(self.hidden_dim),
                    ReLU(),
                    Dropout(self.dropout),)
                self.conv = GINEConv(nn)
    
And then gives the implementation:

.. code-block:: python

    class BaseModel(torch.nn.Module):
        def forward(self, x, edge_index, batch, **kwargs):
            if self.base_model == $New_method:
                x = self.conv(x, edge_index, kwargs['edge_attr'])
            # an example could be:
            if self.base_model in ['GINE', 'DIGINE']:
                x = self.conv(x, edge_index, kwargs['edge_attr'])



Once the base_model.py is editied, the base model could be called by a single line in the general_config as described in `Select from an Existing Method <method_select.html>`_, and flexibly combine with the Positional Encoding (PE) methods with PE configurations.

 


import torch
from torch import nn
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GPSConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric_signed_directed.nn import MSConv
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from models.base_model import MLPs
from models.middle_model import MiddleModel

class TIMEModel(nn.Module):
    def __init__(self, args):
        super(TIMEModel, self).__init__()
        self.args = args
        self.pe_type = args.get('pe_type')
        self.hidden_dim = args['hidden_dim']

        if self.pe_type is not None and args['pe_strategy'] == 'variant':
            pe_dim_output = args['mag_pe_dim_output'] if self.pe_type == 'maglap' else args['lap_pe_dim_output']
            edge_emb_dim = (args['hidden_dim']+pe_dim_output)
            self.pe_variant_layer = Linear(self.hidden_dim + pe_dim_output, self.hidden_dim)
        else:
            edge_emb_dim = (args['hidden_dim'])
        self.node_encoder = Sequential(Linear(10, self.hidden_dim), ReLU(), Linear(self.hidden_dim, self.hidden_dim))
        self.net_edge_encoder = Sequential(Linear(2, edge_emb_dim), ReLU(), Linear(edge_emb_dim, edge_emb_dim))
        self.cell_edge_encoder = Sequential(Linear(8, edge_emb_dim), ReLU(), Linear(edge_emb_dim, edge_emb_dim))
        # define node, edge encoder

        # define middle model
        self.middle_model1 = MiddleModel(self.args)
        self.middle_model2 = MiddleModel(self.args)

        # define final layer
        self.output_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 1, args['mlp_out']['num_layer'])
        self.net_output_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 4, args['mlp_out']['num_layer'])
        self.cell_output_mlp = MLPs(self.middle_model2.out_dims, self.middle_model2.out_dims, 4, args['mlp_out']['num_layer'])

    def forward(self, batch_data):
        #pre-process the node, edge data
        x = self.node_encoder(batch_data.x)
        net_edge_attr = self.net_edge_encoder(batch_data.net_edge_attr)
        cell_edge_attr = self.cell_edge_encoder(batch_data.cell_edge_attr)
        # in the first layer, call the middle model to only message-passing through net edge indices
        if self.pe_type is None:
            x = self.middle_model1(x, batch_data.net_edge_index, batch_data.batch, edge_attr = net_edge_attr)
        else:
            x = self.middle_model1(x, batch_data.net_edge_index, batch_data.batch, edge_attr = net_edge_attr, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        # in the 2nd layer, call another middle model to do message passing through the entire net edge indices
        full_edge_index = torch.cat((batch_data.net_edge_index, batch_data.cell_edge_index), 1)
        full_edge_attr = torch.cat((net_edge_attr, cell_edge_attr), 0)
        if self.pe_type is None:
            x = self.middle_model2(x, full_edge_index, batch_data.batch, edge_attr = full_edge_attr)
        elif self.args['pe_strategy'] == 'variant':
            x = self.pe_variant_layer(x)
            x = self.middle_model1(x, batch_data.net_edge_index, batch_data.batch, edge_attr = net_edge_attr, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        else:
            x = self.middle_model2(x, full_edge_index, batch_data.batch, edge_attr = full_edge_attr, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        #final MLP
        pred = self.output_mlp(x)
        net_delay = self.net_output_mlp(x)
        x_for_cell_delay = x[batch_data.directed_cell_edge_index[0]] + x[batch_data.directed_cell_edge_index[1]]
        cell_delay = self.cell_output_mlp(x_for_cell_delay)
        return pred, net_delay, cell_delay
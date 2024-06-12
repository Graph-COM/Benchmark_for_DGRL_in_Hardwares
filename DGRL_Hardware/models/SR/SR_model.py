


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


class SRModel(nn.Module):
    def __init__(self, args, target):
        super(SRModel, self).__init__()
        self.target = target
        self.args = args
        self.pe_type = args.get('pe_type')
        self.hidden_dim = args['hidden_dim']

        # define node, edge encoder
        self.node_encoder = Sequential(Linear(4, self.hidden_dim//2), ReLU(), Linear(self.hidden_dim//2, self.hidden_dim))
        
        # define middle model
        self.middle_model = MiddleModel(self.args)
        
        # define final layer
        if self.target == 'shared':
            self.output_mlp1 = MLPs(self.middle_model.out_dims, self.middle_model.out_dims, int(args['num_class']), args['mlp_out']['num_layer'])
            self.output_mlp2 = MLPs(self.middle_model.out_dims, self.middle_model.out_dims, int(args['num_class']), args['mlp_out']['num_layer'])
        elif self.target == 'root':
            self.output_mlp = MLPs(self.middle_model.out_dims, self.middle_model.out_dims, int(args['num_class']), args['mlp_out']['num_layer'])

    def forward(self, batch_data):
        #pre-process the node, edge data
        x = self.node_encoder(batch_data.x)
        if self.pe_type is None:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch)
        else:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch,
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        #final MLP
        if self.target == 'shared':
            shared1 = self.output_mlp1(x)
            shared2 = self.output_mlp2(x)
            return shared1, shared2
        elif self.target == 'root':
            root = self.output_mlp(x)
            return root
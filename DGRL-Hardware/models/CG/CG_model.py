 


import torch
from torch import nn
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, ReLU, Sequential, Dropout

from torch_geometric.nn import GINEConv, GCNConv, GPSConv
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.nn.attention import PerformerAttention

from torch_geometric_signed_directed.nn import MSConv
from torch_geometric_signed_directed.nn.directed.complex_relu import complex_relu_layer

from models.base_model import MLPs
from models.middle_model import MiddleModel


class NodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()
        self.node_embedding_list = torch.nn.ModuleList()
        feature_dim_list = [16, 4, 2500, 225, 2500, 1, 60, 60, 2500, 2500, 3, 3, 4, 2500, 225, 2500, 1]
        for dim in feature_dim_list:
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.node_embedding_list.append(emb)
    def forward(self, x):
        x_embedding_list = []
        for i in range(x.shape[1]):
            x_embedding_list.append(self.node_embedding_list[i](x[:,i]))
        x_embedding = torch.cat(x_embedding_list, dim = 1)
        return x_embedding

class CGModel(nn.Module):
    def __init__(self, args, target):
        super(CGModel, self).__init__()
        self.target = target
        self.args = args
        self.pe_type = args.get('pe_type')
        # define node, edge encoder
        node_emb_dim = args['hidden_dim'] // 17
        self.node_encoder = NodeEncoder(node_emb_dim)

        # define middle model
        self.middle_model = MiddleModel(self.args)

        # define final layer
        self.output_mlp = MLPs(self.middle_model.out_dims, self.middle_model.out_dims, 1, args['mlp_out']['num_layer'])

    def forward(self, batch_data):
        #pre-process the node, edge data
        x = self.node_encoder(batch_data.x)
        #call the middle model to process the data
        if self.pe_type is None:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch)
        else:
            x = self.middle_model(x, batch_data.edge_index, batch_data.batch, 
                                  mag_pe = getattr(batch_data, 'mag_pe', None), lap_pe = getattr(batch_data, 'lap_pe', None),
                                  Lambda = batch_data.Lambda)
        x = global_add_pool(x, batch_data.batch)
        x = self.output_mlp(x)
        return x
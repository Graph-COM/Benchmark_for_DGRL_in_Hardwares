from tqdm import tqdm
import os
import random
import pickle
import numpy as np
from easydict import EasyDict
import os.path as osp
import pandas as pd
import re

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

from utils import create_nested_folder
from maglap.get_mag_lap import AddMagLaplacianEigenvectorPE, AddLaplacianEigenvectorPE


class SRDataProcessor(InMemoryDataset):
    def __init__(self, config, mode):
        self.config = config
        self.save_folder = str(config['task']['processed_folder'])+str(config['task']['name'])+'/'+str(config['task']['type'])+'/'
        create_nested_folder(self.save_folder)
        self.divide_seed = config['task']['divide_seed']
        self.mode = mode
        self.raw_data_root = config['task']['raw_data_path']
        self.pe_type = config['model'].get('pe_type')
        if self.pe_type is None:
            pre_transform = None
        elif self.pe_type == 'lap':
            pre_transform = Compose([T.AddRandomWalkPE(walk_length = config['model']['se_pe_dim_input'], attr_name = 'rw_se')])
            self.lap_pre_transform = Compose([AddLaplacianEigenvectorPE(k=config['model']['lap_pe_dim_input'], attr_name='lap_pe')])
        elif self.pe_type == 'maglap':
            pre_transform = Compose([T.AddRandomWalkPE(walk_length = config['model']['se_pe_dim_input'], attr_name = 'rw_se')])
            self.mag_pre_transform = Compose([AddMagLaplacianEigenvectorPE(k=config['model']['mag_pe_dim_input'], q=config['model']['q'],
                                                         multiple_q=config['model']['q_dim'], attr_name='mag_pe')])
        super().__init__(root = self.save_folder, pre_transform = pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[mode])

    @property
    def processed_dir(self) -> str:
        processed_dir = osp.join(self.save_folder, 'processed')
        if self.pe_type is None:
            processed_dir += '_no_pe'
        if self.pe_type == 'lap':
            processed_dir += '_' + self.pe_type + str(self.config['model']['lap_pe_dim_input'])
        elif self.pe_type == 'maglap':
            processed_dir += '_' + str(self.config['model']['mag_pe_dim_input']) + 'k_' + str(self.config['model']['q_dim']) + 'q' + str(self.config['model']['q'])
        return processed_dir
    
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        '''return {
            'train': 'train.pt',
            'valid': 'val.pt',
            'test': 'test.pt',
        }'''
        return {
            'train': 'train.pt',
            'valid': 'val.pt',
            'test': 'test.pt',
            'test_mapped': 'test_mapped.pt',
        }
    @property
    def processed_paths(self):
        return {mode: os.path.join(self.processed_dir, fname) for mode, fname in self.processed_file_names.items()}
    def process(self):
        file_names = self.processed_file_names
        # check if has already created
        exist_flag = 0
        for key in file_names:        
            exist_flag = exist_flag + os.path.isfile(self.processed_paths[key])
        if exist_flag == len(file_names):
            print('all datasets already exists, directly load.')
            return
        else:
            train_path = self.raw_data_root+'/train/'
            valid_path = self.raw_data_root+'/valid/'
            test_path = self.raw_data_root+'/test/'
            test_mapped_path = self.raw_data_root+'/test_mapped/'
            test_list = [32]
            train_graph_list = self.read_csv_graph_raw(train_path)
            valid_graph_list = self.read_csv_graph_raw(valid_path)
            
            test_graph_list = []
            test_mapped_graph_list = []
            for test_id in test_list:
                tmp_test_path = test_path + str(test_id) + '/'
                tmp_test_graph = self.read_csv_graph_raw(tmp_test_path)
                tmp_test_graph = tmp_test_graph[0]
                test_graph_list.append(tmp_test_graph)
                tmp_test_mapped_path = test_mapped_path + str(test_id) + '/'
                tmp_test_mapped_graph = self.read_csv_graph_raw(tmp_test_mapped_path)
                tmp_test_mapped_graph = tmp_test_mapped_graph[0]
                test_mapped_graph_list.append(tmp_test_mapped_graph)
            
            
            for key in file_names:
                data_list = []
                if key == 'train':
                    data = Data(x = torch.tensor(train_graph_list[0]['node_feat']).float(), edge_index = torch.tensor(train_graph_list[0]['edge_index']),
                                shared = torch.tensor(train_graph_list[0]['shared']).long(),
                                root = torch.tensor(train_graph_list[0]['root']).long())
                    # add undirected random walk SE
                    if self.pe_type is not None:
                        bi_edge_index = to_undirected(torch.tensor(train_graph_list[0]['edge_index']))
                        tmp_bidirect_data = Data(x = data.x, edge_index = bi_edge_index) 
                        tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                        data['rw_se'] = tmp_bidirect_data['rw_se']
                        if self.pe_type == 'lap':
                            lap_data = self.lap_pre_transform(data)
                            data['lap_pe'] = lap_data['lap_pe']
                            data['Lambda'] = lap_data['Lambda']
                        elif self.pe_type == 'maglap':
                            mag_data = self.mag_pre_transform(data)
                            data['mag_pe'] = mag_data['mag_pe']
                            data['Lambda'] = mag_data['Lambda']
                    data_list.append(data)
                elif key == 'valid':
                    data = Data(x = torch.tensor(valid_graph_list[0]['node_feat']).float(), edge_index = torch.tensor(valid_graph_list[0]['edge_index']),
                                shared = torch.tensor(valid_graph_list[0]['shared']).long(),
                                root = torch.tensor(valid_graph_list[0]['root']).long())
                    # add undirected random walk SE
                    if self.pe_type is not None:
                        bi_edge_index = to_undirected(torch.tensor(valid_graph_list[0]['edge_index']))
                        tmp_bidirect_data = Data(x = data.x, edge_index = bi_edge_index) 
                        tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                        data['rw_se'] = tmp_bidirect_data['rw_se']
                        if self.pe_type == 'lap':
                            lap_data = self.lap_pre_transform(data)
                            data['lap_pe'] = lap_data['lap_pe']
                            data['Lambda'] = lap_data['Lambda']
                        elif self.pe_type == 'maglap':
                            mag_data = self.mag_pre_transform(data)
                            data['mag_pe'] = mag_data['mag_pe']
                            data['Lambda'] = mag_data['Lambda']
                    data_list.append(data)
                elif key == 'test':
                    for id in range(1):
                        data = Data(x = torch.tensor(test_graph_list[id]['node_feat']).float(), edge_index = torch.tensor(test_graph_list[id]['edge_index']),
                                    shared = torch.tensor(test_graph_list[id]['shared']).long(),
                                    root = torch.tensor(test_graph_list[id]['root']).long())
                        # add undirected random walk SE
                        if self.pe_type is not None:
                            bi_edge_index = to_undirected(torch.tensor(test_graph_list[0]['edge_index']))
                            tmp_bidirect_data = Data(x = data.x, edge_index = bi_edge_index) 
                            tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                            data['rw_se'] = tmp_bidirect_data['rw_se']
                            if self.pe_type == 'lap':
                                lap_data = self.lap_pre_transform(data)
                                data['lap_pe'] = lap_data['lap_pe']
                                data['Lambda'] = lap_data['Lambda']
                            elif self.pe_type == 'maglap':
                                mag_data = self.mag_pre_transform(data)
                                data['mag_pe'] = mag_data['mag_pe']
                                data['Lambda'] = mag_data['Lambda']
                        data_list.append(data)
                elif key == 'test_mapped':
                    for id in range(1):
                        data = Data(x = torch.tensor(test_mapped_graph_list[id]['node_feat']).float(), edge_index = torch.tensor(test_mapped_graph_list[id]['edge_index']),
                                    shared = torch.tensor(test_mapped_graph_list[id]['shared']).long(),
                                    root = torch.tensor(test_mapped_graph_list[id]['root']).long())
                        # add undirected random walk SE
                        if self.pe_type is not None:
                            bi_edge_index = to_undirected(torch.tensor(test_mapped_graph_list[0]['edge_index']))
                            tmp_bidirect_data = Data(x = data.x, edge_index = bi_edge_index) 
                            tmp_bidirect_data = self.pre_transform(tmp_bidirect_data)
                            data['rw_se'] = tmp_bidirect_data['rw_se']
                            if self.pe_type == 'lap':
                                lap_data = self.lap_pre_transform(data)
                                data['lap_pe'] = lap_data['lap_pe']
                                data['Lambda'] = lap_data['Lambda']
                            elif self.pe_type == 'maglap':
                                mag_data = self.mag_pre_transform(data)
                                data['mag_pe'] = mag_data['mag_pe']
                                data['Lambda'] = mag_data['Lambda']
                        data_list.append(data)
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[key])
    def read_csv_graph_raw(self, raw_dir):
        label_dir = raw_dir
        labels_shared = pd.read_csv(osp.join(label_dir, 'node-label-shared.csv'), header = None).values.reshape(-1)
        labels_root = pd.read_csv(osp.join(label_dir, 'node-label-root.csv'), header = None).values.reshape(-1)
        try:
            edge = pd.read_csv(osp.join(raw_dir, 'edge.csv'), header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            num_node_list = pd.read_csv(osp.join(raw_dir, 'num-node-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
            num_edge_list = pd.read_csv(osp.join(raw_dir, 'num-edge-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
        except FileNotFoundError:
            raise RuntimeError('No such file')
        try:
            node_feat = pd.read_csv(osp.join(raw_dir, 'node-feat.csv'), header = None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                node_feat = node_feat.astype(np.float32)
        except FileNotFoundError:
            node_feat = None
        #[0 0 0 0]
        #[1 1 1 1]
        print('node feature min'+str(node_feat.min(axis = 0)))
        print('node feat max:'+str(node_feat.max(axis = 0)))
        try:
            edge_feat = pd.read_csv(osp.join(raw_dir, 'edge-feat.csv'), header = None).values
            if 'int' in str(edge_feat.dtype):
                edge_feat = edge_feat.astype(np.int64)
            else:
                edge_feat = edge_feat.astype(np.float32)
        except FileNotFoundError:
            edge_feat = None

        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0
        print('Processing graphs...')
        for graph_id, (num_node, num_edge) in tqdm(enumerate(zip(num_node_list, num_edge_list))):
            graph = dict()
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]
            if edge_feat is not None:
                graph['edge_feat'] = edge_feat[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph['edge_feat'] = None
            num_edge_accum += num_edge
            ### handling node
            if node_feat is not None:
                graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
            else:
                graph['node_feat'] = None

            graph['shared'] = labels_shared
            graph['root'] = labels_root
            graph['num_nodes'] = num_node
            num_node_accum += num_node
            graph_list.append(graph)
        return graph_list
    
    def add_padding(self, data, target_size):
        num_nodes = data.num_nodes
        if num_nodes <= target_size:
            num_nodes_to_add = target_size - num_nodes + 1
            extra_node_features = torch.zeros((num_nodes_to_add, data.x.shape[1])).long()
            data.x = torch.cat([data.x, extra_node_features], dim=0)
        return data
    

    
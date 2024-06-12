from types import CellType
from tqdm import tqdm
import os
import random
import pickle
import numpy as np
from easydict import EasyDict
import os.path as osp
import pandas as pd
import re
from collections import defaultdict, deque

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

from utils import create_nested_folder
from maglap.get_mag_lap import AddMagLaplacianEigenvectorPE, AddLaplacianEigenvectorPE


class TIMEDataProcessor(InMemoryDataset):
    def __init__(self, config, mode):
        self.config = config
        self.save_folder = str(config['task']['processed_folder'])+str(config['task']['name'])+'/'
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
        if mode == 'train' or mode == 'valid' or mode == 'test_id':
            self.data, self.slices = torch.load(self.processed_paths['id'])
        elif mode == 'test_ood':
            self.data, self.slices = torch.load(self.processed_paths['ood'])
    @property
    def raw_file_names(self):
        return []
    
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
    def processed_file_names(self):
        return {
            'id': 'id.pt',
            'ood': 'ood.pt',
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
            
            #id_file_names = ['blabla', 'usb_cdc_core', 'BM64', 'wbqspiflash', 'cic_decimator', 'picorv32a', 'zipdiv', 'genericfir', 'usb']
            id_file_names = ['blabla', 'usb_cdc_core', 'wbqspiflash', 'cic_decimator', 'picorv32a', 'zipdiv', 'usb']
            ood_file_names = ['xtea', 'spm', 'y_huff', 'synth_ram']
            raw_data_path = self.config['task']['raw_data_path']
            id_graph_list = self.read_csv_graph_raw(raw_data_path, id_file_names)
            ood_graph_list = self.read_csv_graph_raw(raw_data_path, ood_file_names)
            
            for key in tqdm(file_names):
                data_list = []
                if key == 'id':
                    for id in tqdm(range(len(id_graph_list))):
                        endpt_mask = id_graph_list[id]['endpt_mask']
                        endpt_indices = np.where(endpt_mask == 1)[0]
                        shuffled_endpt_indices = np.random.permutation(endpt_indices)
                        num_training = int(len(shuffled_endpt_indices) * 0.45)
                        num_validation = int(len(shuffled_endpt_indices) * 0.35)
                        training_endpt_indices = shuffled_endpt_indices[:num_training]
                        validation_index = shuffled_endpt_indices[num_training:num_training + num_validation]
                        test_index = shuffled_endpt_indices[num_training + num_validation:]
                        # train: 0 test: 1 valid: 2
                        endpt_mask[training_endpt_indices] = 0
                        endpt_mask[validation_index] = 2
                        endpt_mask[test_index] = 1
                        full_edge_index = torch.cat((torch.tensor(id_graph_list[id]['net_edge_index']), torch.tensor(id_graph_list[id]['cell_edge_index'])), 1)
                        data = Data(x = torch.tensor(id_graph_list[id]['node_feat']).to(dtype=torch.float32), 
                                    net_edge_index = torch.tensor(id_graph_list[id]['net_edge_index']), cell_edge_index = torch.tensor(id_graph_list[id]['cell_edge_index']),
                                    edge_index = full_edge_index,
                                    net_edge_attr = torch.tensor(id_graph_list[id]['net_edge_attr']).to(dtype=torch.float32), 
                                    cell_edge_attr = torch.tensor(id_graph_list[id]['cell_edge_attr']).to(dtype=torch.float32), 
                                    cell_delay = torch.tensor(id_graph_list[id]['cell_delay']).to(dtype=torch.float32),
                                    net_delay = torch.tensor(id_graph_list[id]['net_delay']).to(dtype=torch.float32),
                                    hold = torch.tensor(id_graph_list[id]['hold']).reshape(-1, 1).to(dtype=torch.float32),
                                    setup = torch.tensor(id_graph_list[id]['setup']).reshape(-1, 1).to(dtype=torch.float32),
                                    endpt_mask = torch.tensor(endpt_mask).long())
                        
                        # add undirected random walk SE
                        if self.pe_type is not None:
                            bi_edge_index = to_undirected(full_edge_index)
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
                elif key == 'ood':
                    for id in tqdm(range(len(ood_graph_list))):
                        endpt_mask = ood_graph_list[id]['endpt_mask']
                        full_edge_index = torch.cat((torch.tensor(ood_graph_list[id]['net_edge_index']), torch.tensor(ood_graph_list[id]['cell_edge_index'])), 1)
                        data = Data(x = torch.tensor(ood_graph_list[id]['node_feat']).to(dtype=torch.float32), 
                                    net_edge_index = torch.tensor(ood_graph_list[id]['net_edge_index']), cell_edge_index = torch.tensor(ood_graph_list[id]['cell_edge_index']),
                                    edge_index = full_edge_index,
                                    net_edge_attr = torch.tensor(ood_graph_list[id]['net_edge_attr']).to(dtype=torch.float32), 
                                    cell_edge_attr = torch.tensor(ood_graph_list[id]['cell_edge_attr']).to(dtype=torch.float32),
                                    cell_delay = torch.tensor(ood_graph_list[id]['cell_delay']).to(dtype=torch.float32),
                                    net_delay = torch.tensor(ood_graph_list[id]['net_delay']).to(dtype=torch.float32), 
                                    hold = torch.tensor(ood_graph_list[id]['hold']).reshape(-1, 1).to(dtype=torch.float32),
                                    setup = torch.tensor(ood_graph_list[id]['setup']).reshape(-1, 1).to(dtype=torch.float32),
                                    endpt_mask = torch.tensor(endpt_mask).long())
                        
                        # add undirected random walk SE
                        if self.pe_type is not None:
                            bi_edge_index = to_undirected(full_edge_index)
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
    def read_csv_graph_raw(self, raw_dir, file_name_list):
        graph_list = []
        for file_name in file_name_list:
            file_dir = raw_dir + file_name + '/raw/'
            label_hold_path = file_dir + 'node-label-hold-slack.csv'
            label_setup_path = file_dir + 'node-label-setup-slack.csv'
            cell_delay_path = file_dir + 'celldelay.csv'
            net_delay_path = file_dir + 'netdelay_log.csv'
            label_hold = pd.read_csv(label_hold_path, header = None).values.T.astype(np.float32) # 1*N
            label_setup = pd.read_csv(label_setup_path, header = None).values.T.astype(np.float32) # 1*N
            cell_delay = pd.read_csv(cell_delay_path, header = None).values.astype(np.float32)
            net_delay = pd.read_csv(net_delay_path, header = None).values.astype(np.float32)
            
            all_edge_index = pd.read_csv(osp.join(file_dir, 'edge.csv'), header = None).values.T.astype(np.int64)
            '''has_cycle = self.has_cycle(all_edge_index)
            print(has_cycle)
            topo_order, node_order = self.topological_sort_with_order(all_edge_index)
            print(topo_order)
            import pdb; pdb.set_trace()'''
            num_node_list = pd.read_csv(osp.join(file_dir, 'num-node-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
            num_edge_list = pd.read_csv(osp.join(file_dir, 'num-edge-list.csv'), header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
        
            node_feat = pd.read_csv(osp.join(file_dir, 'node-feat.csv'), header = None).values
            node_feat = node_feat.astype(np.float32)

            edge_feat_all = pd.read_csv(osp.join(file_dir, 'edge-feat.csv'), header = None).values
            edge_feat_all = edge_feat_all.astype(np.float32)

            # end point mask
            end_point_mask = pd.read_csv(osp.join(file_dir, 'is-timing-endpt.csv'), header = None).values

            # get the net edge and cell edge, and there edge features, separatedly
            net_edge_num = np.where((edge_feat_all[:, 0] == 0) & (edge_feat_all[:, 1] == 1))[0]
            cell_edge_num = np.where((edge_feat_all[:, 0] == 1) & (edge_feat_all[:, 1] == 0))[0]

            net_edge_attr = edge_feat_all[net_edge_num, :][:, 10:12]
            cell_edge_attr = edge_feat_all[cell_edge_num, :][:, 2:10]

            net_edge_index = all_edge_index[:, net_edge_num]
            cell_edge_index = all_edge_index[:, cell_edge_num]
            # allocate the graph
            graph = dict()
            graph['net_edge_index'] = net_edge_index
            graph['cell_edge_index'] = cell_edge_index
            graph['net_edge_attr'] = net_edge_attr
            graph['cell_edge_attr'] = cell_edge_attr
            graph['cell_delay'] = cell_delay
            graph['net_delay'] = net_delay


            graph['node_feat'] = node_feat
            graph['hold'] = label_hold
            graph['setup'] = label_setup
            graph['endpt_mask'] = end_point_mask
            graph_list.append(graph)
        return graph_list
    

    def has_cycle(self, edge_index):
        from collections import defaultdict

        def dfs(node, graph, visited, rec_stack):
            if not visited[node]:
                visited[node] = True
                rec_stack[node] = True

                for neighbour in graph[node]:
                    if not visited[neighbour] and dfs(neighbour, graph, visited, rec_stack):
                        return True
                    elif rec_stack[neighbour]:
                        return True

            rec_stack[node] = False
            return False

        graph = defaultdict(list)
        for src, dst in zip(edge_index[0], edge_index[1]):
            graph[src].append(dst)

        visited = defaultdict(bool)
        rec_stack = defaultdict(bool)

        for node in set(edge_index[0] + edge_index[1]):
            if not visited[node] and dfs(node, graph, visited, rec_stack):
                return True  # Found at least one cycle

        return False  # No cycles found
    
        
    def topological_sort_with_order(self, edge_index):
        # 创建图的邻接表和入度表
        adj_list = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set(edge_index[0] + edge_index[1])

        # 填充邻接表和入度表
        for src, dst in zip(edge_index[0], edge_index[1]):
            adj_list[src].append(dst)
            in_degree[dst] += 1
            if src not in in_degree:
                in_degree[src] = 0

        # 使用队列找到所有入度为0的节点
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        topo_order = []  # 存储拓扑排序的结果
        node_order = {}  # 存储每个节点的拓扑顺序

        order = 0  # 初始化拓扑顺序
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            node_order[node] = order
            order += 1
            for neighbor in adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        #if len(topo_order) == len(all_nodes):
        return topo_order, node_order  # 返回拓扑排序和每个节点的拓扑顺序
        #else:
            #return "Graph has a cycle, topological sort not possible.", {}
    
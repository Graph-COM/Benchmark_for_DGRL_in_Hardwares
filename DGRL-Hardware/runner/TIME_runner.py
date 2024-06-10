import numpy as np
import wandb
import shutil
import yaml
import importlib
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import r2_score

import torch

from torchmetrics.regression import MeanAbsolutePercentageError

from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch_geometric.utils import to_undirected

import ray
from ray import train, tune
from ray.train import Checkpoint, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from hyperopt import hp

from utils import exception_not_defined, create_nested_folder, find_latest_model, delete_file_with_head

class TIMERunner():
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.train_folder = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/'
        self.result_csv = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/result.csv'
        create_nested_folder(self.train_folder)
        # define the loss criterion
        if self.config['train']['criterion'] == 'L1':
            self.criterion = nn.L1Loss()
        elif self.config['train']['criterion'] == 'SmoothL1':  
            self.criterion = nn.SmoothL1Loss()
        elif self.config['train']['criterion'] == 'MSE':
            self.criterion = nn.MSELoss()
        else: 
            exception_not_defined('criterion')
        self.mse = nn.MSELoss()
        self.mape = MeanAbsolutePercentageError()
    def train_ray(self, tune_parameter_config):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        batch_size = 1
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)

        # define the model, optimizer, and scheduler here
        module_path = 'models.'+self.config['task']['name'] + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name'] + 'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(tune_parameter_config)
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(tune_parameter_config['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        #self.device = self.config['train']['device']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.model.train()
        self.best_valid_metric = 10000
        self.mape.to(self.device)

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_mse, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_mse, valid_r2 = self.train_one_epoch(self.train_loader, 'valid', epoch_idx)
            self.scheduler.step()
            self.save_model(valid_mse, epoch_idx)
            train.report({'hidden_dim': tune_parameter_config['hidden_dim'], 'num_layer': tune_parameter_config['num_layers'], 
                          'mse' : valid_mse, 'r2' : valid_r2,
                          'lr': tune_parameter_config['lr'],'dropout': tune_parameter_config['dropout'], 
                          'mlp_out': tune_parameter_config['mlp_out']['num_layer'],
                        })
    
    def train(self):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        batch_size = self.config['train']['batch_size']
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size, shuffle = True)
        
        # define the model, optimizer, and scheduler here
        module_path = 'models.'+str(self.config['task']['name']) + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name']+'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(self.config['model'])
        #self.model = cls(self.config['model']['args'])
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['train']['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        self.device = int(self.config['train']['device'])
        self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
        self.model.train()
        self.best_valid_metric = 10000
        self.mape.to(self.device)

        # initialize the normalization of regression

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_mse, train_r2 = self.train_one_epoch(self.train_loader, 'train', epoch_idx)
            valid_loss, valid_mse, valid_r2 = self.train_one_epoch(self.train_loader, 'valid', epoch_idx)
            self.scheduler.step()
            self.save_model(valid_mse, epoch_idx)

    def train_one_epoch(self, data_loader, mode, epoch_idx):
        epoch_loss = 0
        epoch_mse = 0
        epoch_pred_mse = 0
        epoch_cell_mse = 0
        epoch_net_mse = 0
        epoch_r2 = 0
        total_nodes = 0
        total_graphs = 0
        for batch_data in data_loader:
            batch_data.to(self.device)
            net_edge_index = batch_data.net_edge_index
            cell_edge_index = batch_data.cell_edge_index
            net_edge_attr = batch_data.net_edge_attr
            cell_edge_attr = batch_data.cell_edge_attr
            batch_data.directed_cell_edge_index = cell_edge_index
            if self.config['train']['directed'] == 0:
                net_edge_index, net_edge_attr = to_undirected(net_edge_index, net_edge_attr, reduce = 'add')
                cell_edge_index, cell_edge_attr = to_undirected(cell_edge_index, cell_edge_attr, reduce = 'add')
                batch_data.net_edge_index = net_edge_index
                batch_data.cell_edge_index = cell_edge_index
                batch_data.net_edge_attr = net_edge_attr
                batch_data.cell_edge_attr = cell_edge_attr
            y = getattr(batch_data, self.config['task']['target'], None)
            if mode == 'train':
                self.model.train()
                self.optimizer.zero_grad()
                if self.config['model']['name'] in ['GPS', 'GPSSE']:
                    self.model.middle_model1.redraw_projection.redraw_projections()
                    self.model.middle_model2.redraw_projection.redraw_projections()
                    pred, net_delay, cell_delay = self.model(batch_data)
                else:
                    pred, net_delay, cell_delay = self.model(batch_data)
                pred_wanted = pred[torch.where(batch_data.endpt_mask==0)[0]]
                y_wanted = y[torch.where(batch_data.endpt_mask==0)[0]]
            elif mode == 'valid':
                #self.model.eval()
                with torch.no_grad():
                    if self.config['model']['name'] in ['GPS', 'GPSSE']:
                        pred, net_delay, cell_delay = self.model(batch_data)
                    else:
                        pred, net_delay, cell_delay = self.model(batch_data)
                pred_wanted = pred[torch.where(batch_data.endpt_mask==2)[0]]
                y_wanted = y[torch.where(batch_data.endpt_mask==2)[0]]
            
            pred_loss = self.criterion(pred_wanted, y_wanted.reshape(-1, 1))
            net_loss = self.criterion(net_delay, batch_data.net_delay)
            cell_loss = self.criterion(cell_delay, batch_data.cell_delay)
            batch_loss = pred_loss + net_loss + cell_loss
            pred_mse = self.mse(pred_wanted, y_wanted.reshape(-1, 1))
            net_mse = self.mse(net_delay, batch_data.net_delay)
            cell_mse = self.mse(cell_delay, batch_data.cell_delay)
            batch_mse = pred_mse + net_mse + cell_mse
            batch_r2 = r2_score(y_wanted.detach().cpu().numpy().reshape(-1), pred_wanted.detach().cpu().numpy().reshape(-1))
            if mode == 'train':
                batch_loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss + batch_loss.item() * (batch_data.x.shape[0])
            epoch_mse = epoch_mse + batch_mse.item() * (batch_data.x.shape[0])
            epoch_pred_mse = epoch_pred_mse + pred_mse.item() * (batch_data.x.shape[0])
            epoch_net_mse = epoch_net_mse + net_mse.item() * (batch_data.x.shape[0])
            epoch_cell_mse = epoch_cell_mse + cell_mse.item() * (batch_data.x.shape[0])
            epoch_r2 = epoch_r2 + batch_r2
            total_nodes = total_nodes + (batch_data.x.shape[0])
            total_graphs = total_graphs + 1
        epoch_loss = epoch_loss / total_nodes
        epoch_pred_mse = epoch_pred_mse / total_nodes
        epoch_net_mse = epoch_net_mse / total_nodes
        epoch_cell_mse = epoch_cell_mse / total_nodes
        epoch_mse = epoch_mse / total_nodes
        epoch_r2 = epoch_r2 / total_graphs
        self.write_log({'loss': epoch_loss, 'total mse': epoch_mse,
                        'pred mse': epoch_pred_mse, 
                        'net mse': epoch_net_mse, 'cell mse': epoch_cell_mse, 'r2': epoch_r2}, epoch_idx, mode)
        return epoch_loss, epoch_mse, epoch_r2
     
    def test(self, load_statedict = True, test_num_idx = 0):
        if load_statedict:
            self.device = int(self.config['train']['device'])
            module_path = 'models.'+str(self.config['task']['name']) + '.' + self.config['task']['name']+'_model'
            attribute_name = self.config['task']['name']+'Model'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            self.model = cls(self.config['model'])
            dict_path = find_latest_model(self.train_folder, 'model')
            state_dict = torch.load(dict_path) 
            self.model.load_state_dict(state_dict)
            self.model = self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
            #self.model.eval()
        else:
            pass
            #self.model.eval() 
        # start testing
        test_list = ['id', 'ood']
        self.test_data_dict = {}
        self.test_loader_dict = {}
        for test_name in test_list:
            module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
            attribute_name = self.config['task']['name']+'DataProcessor'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            test_data = cls(self.config, 'test_'+test_name)
            self.test_data_dict[test_name] = test_data
            test_loader = DataLoader(test_data, batch_size = 1, shuffle = False)
            self.test_loader_dict[test_name] = test_loader
        id_file_names = ['blabla', 'usb_cdc_core', 'wbqspiflash', 'cic_decimator', 'picorv32a', 'zipdiv', 'usb']
        ood_file_names = ['xtea', 'spm', 'y_huff', 'synth_ram']
        table = PrettyTable(['test set name', '# of samples', 'mse loss', 'r2 score'])
        for test_name in test_list:
            mse_list, r2_list = self.test_a_task(test_name)
            if test_name == 'id':
                for circuit_idx, id_circuit in enumerate(id_file_names):
                    row = [str(id_circuit), '1', mse_list[circuit_idx], r2_list[circuit_idx]]
                    table.add_row(row)
                row = ['id avg', str(len(id_file_names)), np.mean(mse_list), np.mean(r2_list)]
                table.add_row(row)
            elif test_name == 'ood':
                for circuit_idx, ood_circuit in enumerate(ood_file_names):
                    row = [str(ood_circuit), '1', mse_list[circuit_idx], r2_list[circuit_idx]]
                    table.add_row(row)
                row = ['ood avg', str(len(ood_file_names)), np.mean(mse_list), np.mean(r2_list)]
                table.add_row(row)
        if test_num_idx == 0:
            with open(self.result_csv, 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
        else:
            with open(self.result_csv, 'a', newline='') as f_output:
                f_output.write(table.get_csv_string())
        print(table)
        
    def test_a_task(self, testset_name):
        mse_list = []
        r2_list = []
        for data_idx, batch_data in tqdm(enumerate(self.test_loader_dict[testset_name])):
            batch_data.to(self.device)
            net_edge_index = batch_data.net_edge_index
            cell_edge_index = batch_data.cell_edge_index
            net_edge_attr = batch_data.net_edge_attr
            cell_edge_attr = batch_data.cell_edge_attr
            batch_data.directed_cell_edge_index = cell_edge_index
            if self.config['train']['directed'] == 0:
                net_edge_index, net_edge_attr = to_undirected(net_edge_index, net_edge_attr, reduce = 'add')
                cell_edge_index, cell_edge_attr = to_undirected(cell_edge_index, cell_edge_attr, reduce = 'add')
                batch_data.net_edge_index = net_edge_index
                batch_data.cell_edge_index = cell_edge_index
                batch_data.net_edge_attr = net_edge_attr
                batch_data.cell_edge_attr = cell_edge_attr
            y = getattr(batch_data, self.config['task']['target'], None)
            with torch.no_grad():
                pred, net_delay, cell_delay = self.model(batch_data)
            y_test = y[torch.where(batch_data.endpt_mask == 1)[0]]
            pred_test = pred[torch.where(batch_data.endpt_mask == 1)[0]]
            mse = self.mse(y_test.cpu().reshape(-1), pred_test.cpu().reshape(-1)).item()
            r2 = r2_score(y_test.cpu().reshape(-1), pred_test.cpu().reshape(-1)).item()
            mse_list.append(mse)
            r2_list.append(r2)
        return mse_list, r2_list

    def save_config(self):
        with open(self.train_folder + 'config.yaml', 'w') as file:
            yaml.dump(self.config, file)
        return
    def write_log(self, items, epoch_idx, mode):
        print('epoch: '+str(epoch_idx)+' '+str(mode))
        for key in items.keys():
            print(str(key) + ' ' + str(items[key]))
        if self.config['train']['wandb'] == 1:
            for key in items.keys():
                wandb.log({mode+ ' ' + str(key): items[key]}, step = epoch_idx)
    def init_wandb(self):
        if self.config['train']['wandb'] == 1:
            wandb.init(project='EDA_benchmark', name = self.config['task']['name']+'_'+str(self.config['task']['type'])+'_'+self.config['task']['target']+'_'+self.config['model']['name'])
    def save_model(self, valid_metric, epoch_idx):
        if valid_metric < self.best_valid_metric:
            self.best_valid_metric = valid_metric
            delete_file_with_head(self.train_folder, 'model')
            torch.save(self.model.state_dict(), self.train_folder+'model'+'_epoch'+str(epoch_idx)+'.pth')

    def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
        #reporter = CLIReporter(parameter_columns=['hidden_dim', 'num_layer', 'lr', 'dropout', 'mlp_out'],metric_columns=['loss', 'mse', 'r2'])
        reporter = CLIReporter(parameter_columns=['hidden_dim'],metric_columns=['loss', 'mse', 'r2'])
        if self.config['model'].get('pe_file_name') in ['lap_naive', 'maglap_1q_naive'] and self.config['model']['name'] in ['PERFORMER']:
            hidden_dim = 16 + 32 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        else: 
            hidden_dim = 32 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1]))
        # init ray tune
        dropout_p = hp.choice('dropout_p', tune_config['dropout'])
        tune_parameter_config = {
        'name': tune_config['name'],
        'hidden_dim': hidden_dim,
        'num_layers': hp.randint('num_layers', int(tune_config['num_layers'][0]), int(tune_config['num_layers'][1])),
        'lr': hp.uniform('lr', float(tune_config['lr'][0]), float(tune_config['lr'][1])),
        'dropout': dropout_p,
        'node_input_dim': self.config['model']['node_input_dim'],
        'edge_input_dim': self.config['model']['edge_input_dim'],
        'pe_dim_input': tune_config['pe_dim_input'],
        'pe_dim_output': tune_config['pe_dim_output'],
        'mlp_out': {'num_layer': hp.randint('mlp_out', int(tune_config['mlp_out']['num_layer'][0]), 
                                            int(tune_config['mlp_out']['num_layer'][1]))},
        'criterion': 'MSE',
        'attn_type': 'multihead',
        'attn_kwargs': {'dropout': dropout_p},
        }
        tune_parameter_config = {**self.config['model'], **tune_parameter_config}
        scheduler = ASHAScheduler(
            max_t=1000,
            grace_period=500,
            reduction_factor=2)
        
        hyperopt_search = HyperOptSearch(tune_parameter_config, metric='mse', mode='min')

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_ray),
                resources={'cpu': num_cpu, 'gpu': num_gpu_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='mse',
                mode='min',
                scheduler=scheduler,
                num_samples=num_samples,
                search_alg=hyperopt_search,   
            ),
            run_config=RunConfig(progress_reporter=reporter),
        )
        results = tuner.fit()
        
        best_result = results.get_best_result('mse', 'min')

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics['loss']))
        
        print("Best trial final validation mse: {}".format(
            best_result.metrics['pred mse']))
        
        print("Best trial final validation r2: {}".format(
            best_result.metrics['r2']))
        
        r2_result = results.get_best_result('r2', 'min')
        print("Best trial config: {}".format(r2_result.config))
        print("Best trial final validation loss: {}".format(
            r2_result.metrics['loss']))
        
        print("Best trial final validation mse: {}".format(
            r2_result.metrics['pred mse']))
        
        print("Best trial final validation mse: {}".format(
            r2_result.metrics['r2']))
        
        import pdb; pdb.set_trace()

    
        
            
                
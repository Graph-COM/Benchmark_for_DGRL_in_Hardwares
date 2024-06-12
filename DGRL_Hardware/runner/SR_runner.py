import numpy as np
import wandb
import shutil
import yaml
import importlib
from tqdm import tqdm
from prettytable import PrettyTable
import copy

import torch

from torchmetrics.regression import MeanAbsolutePercentageError
from torchmetrics import Precision, Recall, F1Score

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

class SRRunner():
    def __init__(self, config):
        self.config = config
        self.task = config['task']
        self.train_folder = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/'
        self.result_csv = config['train']['train_files']+str(config['task']['name'])+ '_' +str(config['task']['type'])+'/'+str(config['task']['target'])+'_'+str(config['model'].get('pe_file_name'))+'/'+str(config['model']['name'])+'/result.csv'
        create_nested_folder(self.train_folder)
        # define the loss criterion
        if self.config['train']['criterion'] == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else: 
            exception_not_defined('criterion')
        self.num_class = self.config['model']['num_class']
        if self.config['task']['target'] == 'shared':
            self.precision_calculator = Precision(task = 'multiclass', average='macro', num_classes=4)
            self.recall_calculator = Recall(task = 'multiclass', average='macro', num_classes=4)
            self.f1_calculator = F1Score(task = 'multiclass', average='macro', num_classes=4)
        elif self.config['task']['target'] == 'root':
            self.precision_calculator = Precision(task = 'multiclass', average='macro', num_classes=self.num_class)
            self.recall_calculator = Recall(task = 'multiclass', average='macro', num_classes=self.num_class)
            self.f1_calculator = F1Score(task = 'multiclass', average='macro', num_classes=self.num_class)

    def train_ray(self, tune_parameter_config):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.valid_data = cls(self.config, 'valid')
        self.train_loader = DataLoader(self.train_data, batch_size = 1)
        self.val_loader = DataLoader(self.valid_data, batch_size = 1)
        # define the model, optimizer, and scheduler here
        module_path = 'models.'+self.config['task']['name'] + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name'] + 'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(tune_parameter_config, self.config['task']['target'])
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(tune_parameter_config['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.precision_calculator.to(self.device)
        self.recall_calculator.to(self.device)
        self.f1_calculator.to(self.device)

        self.model.to(self.device)
        self.model.train()
        self.best_valid_metric = 10000

        train_y_shared, train_y_shared1, train_y_shared2, train_y_root = self.get_y1(self.train_data[0])
        valid_y_shared, valid_y_shared1, valid_y_shared2, valid_y_root = self.get_y1(self.valid_data[0])

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_one_epoch(self.train_loader, 'train', epoch_idx, train_y_shared, train_y_shared1, train_y_shared2, train_y_root)
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx, valid_y_shared, valid_y_shared1, valid_y_shared2, valid_y_root)
            self.scheduler.step()
            #self.save_model(valid_loss, epoch_idx)
            train.report({'loss' : valid_loss, 'acc' : valid_acc,
                           'precision' : valid_precision,
                            'recall': valid_recall, 'f1': valid_f1})


    def train(self):
        self.init_wandb()
        # initialize the datasets and dataloader
        module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
        attribute_name = self.config['task']['name']+'DataProcessor'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.train_data = cls(self.config, 'train')
        self.valid_data = cls(self.config, 'valid')
        self.train_loader = DataLoader(self.train_data, batch_size = 1)
        self.val_loader = DataLoader(self.valid_data, batch_size = 1)
        # define the model, optimizer, and scheduler here
        module_path = 'models.'+ self.config['task']['name'] + '.' + self.config['task']['name']+'_model'
        attribute_name = self.config['task']['name']+'Model'
        module = importlib.import_module(module_path)
        cls = getattr(module, attribute_name)
        self.model = cls(self.config['model'], self.config['task']['target'])
        if self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.config['train']['lr']), betas=(0.9, 0.999))
        else:
            exception_not_defined('optimizer')
        if self.config['train']['scheduler']['name'] == 'StepLR':
            self.scheduler = StepLR(self.optimizer, step_size=self.config['train']['scheduler']['step_size'], gamma=self.config['train']['scheduler']['gamma'])
        else: 
            exception_not_defined('scheduler')
        
        self.device = int(self.config['train']['device'])
        self.precision_calculator.to(self.device)
        self.recall_calculator.to(self.device)
        self.f1_calculator.to(self.device)

        self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
        self.model.train()
        self.best_valid_metric = 10000
        train_y_shared, train_y_shared1, train_y_shared2, train_y_root = self.get_y1(self.train_data[0])
        valid_y_shared, valid_y_shared1, valid_y_shared2, valid_y_root = self.get_y1(self.valid_data[0])

        for epoch_idx in tqdm(range(self.config['train']['epoch'])):
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.train_one_epoch(self.train_loader, 'train', epoch_idx, train_y_shared, train_y_shared1, train_y_shared2, train_y_root)
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.train_one_epoch(self.val_loader, 'valid', epoch_idx, valid_y_shared, valid_y_shared1, valid_y_shared2, valid_y_root)
            self.scheduler.step()
            self.save_model(valid_loss, epoch_idx)

    def train_one_epoch(self, data_loader, mode, epoch_idx, y_shared, y_shared1, y_shared2, y_root):
        if mode == 'val':
            self.model.eval()
        elif mode == 'train':
            self.model.train()
        for batch_data in data_loader:
            batch_data.to(self.device)
            edge_index = batch_data.edge_index
            if self.config['train']['directed'] == 0:
                edge_index = to_undirected(edge_index)
                batch_data.edge_index = edge_index
            if mode == 'train':
                self.optimizer.zero_grad()
                if self.config['model']['name'] in ['GPS', 'GPSSE']:
                    self.model.middle_model.redraw_projection.redraw_projections()
                    result = self.model(batch_data)
                else:
                    result = self.model(batch_data)
            elif mode == 'valid':
                with torch.no_grad():
                    if self.config['model']['name'] in ['GPS', 'GPSSE']:
                        self.model.middle_model.redraw_projection.redraw_projections()
                        result = self.model(batch_data)
                    else:
                        result = self.model(batch_data)
            if self.config['task']['target'] == 'shared':
                loss_shared_1 = self.criterion(result[0], y_shared1)
                loss_shared_2 = self.criterion(result[1], y_shared2)
                loss = loss_shared_1 + loss_shared_2
                pred_discrete = self.post_processing(result[0].detach(), result[1].detach()).reshape(-1)
                precision, recall, f1 = self.classification_metric(pred_discrete-1, y_shared-1)
            elif self.config['task']['target'] == 'root':
                loss = self.criterion(result, y_root)
                _, pred_discrete = result.detach().max(dim=1)
                precision, recall, f1 = self.classification_metric(pred_discrete, y_root)
            acc = torch.sum(pred_discrete==y_shared) / pred_discrete.shape[0]
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
            epoch_loss = loss.item()
            epoch_precision = precision.item()
            epoch_recall = recall.item()
            epoch_f1 = f1.item()
            epoch_acc = acc.item()
        self.write_log({'loss': epoch_loss, 'accuracy': epoch_acc, 'precision': epoch_precision, 'recall': epoch_recall, 'f1': epoch_f1}, epoch_idx, mode)
        return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1
    
    def get_y1(self, data):
        y_shared = getattr(data, 'shared', None)
        y_shared1, y_shared2 = self.get_processd_y(y_shared)
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1

        y_root = getattr(data, 'root', None)
        # for root classification
        # 0: PO, 1: maj, 2: xor, 3: and, 4: PI
        for i in range(y_root.shape[0]):
            if y_root[i] == 0 or y_root[i] == 4:
                y_root[i] = 3
            y_root[i] = y_root[i] - 1 # 3 classes: 0: maj, 1: xor, 2: and+PI+PO
        return y_shared.to(self.device), y_shared1.to(self.device), y_shared2.to(self.device), y_root.to(self.device)
    
    def get_processd_y(self, y_shared):
        y_shared1 = y_shared.clone().detach()
        y_shared2 = y_shared.clone().detach()

        ### build labels for multitask
        ### original 0: PO, 1: plain, 2: shared, 3: maj, 4: xor, 5: PI
        for i in range(y_shared1.shape[0]):
            if y_shared1[i] == 0 or y_shared1[i] == 5:
                y_shared1[i] = 1
            if y_shared1[i] == 2: 
                y_shared1[i] = 4
            if y_shared1[i] > 2:
                y_shared1[i] = y_shared1[i] - 1 # make to 5 classes
            y_shared1[i] = y_shared1[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
                
        for i in range(y_shared2.shape[0]):
            if y_shared2[i] > 2:
                y_shared2[i] = y_shared2[i] - 1 # make to 5 classes
            if y_shared2[i] == 0 or y_shared2[i] == 4:
                y_shared2[i] = 1
            y_shared2[i] = y_shared2[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
        return y_shared1, y_shared2

    def classification_metric(self, pred, y):
        return self.precision_calculator(pred, y), self.recall_calculator(pred, y),  self.f1_calculator(pred, y)
    
    def post_processing(self, out1, out2):
        pred_1 = out1.argmax(dim=-1, keepdim=True)
        pred_2 = out2.argmax(dim=-1, keepdim=True)
        pred_ecc = (out1 + out2).argmax(dim=-1, keepdim=True)
        
        pred = copy.deepcopy(pred_1)
        
        eq_idx = (torch.eq(pred_1, pred_2) == True).nonzero(as_tuple=True)[0]
        # if pred_1[i] != 0  # maj, xor
        eq_mx_idx = (pred_1[eq_idx] != 0).nonzero(as_tuple=True)[0]
        # pred_1[i] = pred_1[i] + 2  -->  3, 4
        pred[eq_idx[eq_mx_idx]] = pred_1[eq_idx[eq_mx_idx]] + 2
        # if pred_1[i] == 0 PI/PI/and --> final 1
        eq_aig_idx = (pred_1[eq_idx] == 0).nonzero(as_tuple=True)[0]
        pred[eq_idx[eq_aig_idx]] = 1

        neq_idx = (torch.eq(pred_1, pred_2) == False).nonzero(as_tuple=True)[0]
        # if pred_1[i] == 1 and pred_2[i] == 2 shared --> 2
        p1 = (pred_1[neq_idx] == 2).nonzero(as_tuple=True)[0]
        p2 = (pred_2[neq_idx] == 1).nonzero(as_tuple=True)[0]
        shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
        pred[neq_idx[shared]] = 2
        
        p1 = (pred_1[neq_idx] == 1).nonzero(as_tuple=True)[0]
        p2 = (pred_2[neq_idx] == 2).nonzero(as_tuple=True)[0]
        shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
        pred[neq_idx[shared]] = 2
        # else (error correction for discrepant predictions)
        if len(p1) != len(p2) or len(p1) != len(neq_idx):
            v, freq = torch.unique(torch.cat((p1, p2), 0), sorted=True, return_inverse=False, return_counts=True, dim=None)
            uniq = (freq == 1).nonzero(as_tuple=True)[0]
            ecc = v[uniq]
            ecc_mx = (pred_ecc[neq_idx][ecc] != 0).nonzero(as_tuple=True)[0]
            ecc_aig = (pred_ecc[neq_idx][ecc] == 0).nonzero(as_tuple=True)[0]
            pred[neq_idx[ecc[ecc_mx]]] = pred_ecc[neq_idx][ecc][ecc_mx] + 2
            pred[neq_idx[ecc[ecc_aig]]] = 1
            zz = (pred == 0).nonzero(as_tuple=True)[0]
            pred[zz] = 1

        return torch.reshape(pred, (pred.shape[0], 1))  
    
    def test(self, load_statedict = True, test_num_idx = 0):
        if load_statedict:
            self.device = int(self.config['train']['device'])
            module_path = 'models.'+ self.config['task']['name'] + '.' + self.config['task']['name']+'_model'
            attribute_name = self.config['task']['name']+'Model'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            self.model = cls(self.config['model'], self.config['task']['target'])
            dict_path = find_latest_model(self.train_folder, 'model')
            state_dict = torch.load(dict_path) 
            self.model.load_state_dict(state_dict)
            self.model = self.model.to('cuda:'+str(self.device) if torch.cuda.is_available() else 'cpu')
            self.model.eval()
        else:
            self.model.eval() 
        # start testing
        test_list = ['test', 'test_mapped']
        self.test_data_dict = {}
        self.test_loader_dict = {}
        for test_name in test_list:
            module_path = 'data_processor'+'.'+self.config['task']['name']+'_data_processor'
            attribute_name = self.config['task']['name']+'DataProcessor'
            module = importlib.import_module(module_path)
            cls = getattr(module, attribute_name)
            test_data = cls(self.config, test_name)
            self.test_data_dict[test_name] = test_data
            test_loader = DataLoader(test_data, batch_size = 1)
            self.test_loader_dict[test_name] = test_loader
        table = PrettyTable(['test set name', '# of samples', 'accuracy', 'precision', 'recall', 'f1'])
        for test_name in test_list:
            accuracy, precision, recall, f1 = self.test_a_task(test_name)
            row = [str(test_name), str(len(self.test_data_dict[test_name])), accuracy, precision, recall, f1]
            table.add_row(row)
        if test_num_idx == 0:
            with open(self.result_csv, 'w', newline='') as f_output:
                f_output.write(table.get_csv_string())
        else:
            with open(self.result_csv, 'a', newline='') as f_output:
                f_output.write(table.get_csv_string())
        print(table)
        
    def test_a_task(self, testset_name):
        pred_list = []
        y_list = []
        for data_idx, batch_data in tqdm(enumerate(self.test_loader_dict[testset_name])):
            batch_data.to(self.device)
            edge_index = batch_data.edge_index
            if self.config['train']['directed'] == 0:
                edge_index = to_undirected(edge_index)
                batch_data.edge_index = edge_index
            y_shared, y_shared1, y_shared2, y_root = self.get_y1(batch_data)
            with torch.no_grad():
                if self.config['model']['name'] in ['GPS', 'GPSSE']:
                    result = self.model(batch_data)
                else:
                    result = self.model(batch_data)
            if self.config['task']['target'] == 'shared':
                pred_discrete = self.post_processing(result[0].detach(), result[1].detach()).reshape(-1)
                y_list.append(y_shared.reshape(-1, 1))
            elif self.config['task']['target'] == 'root':
                _, pred_discrete = result.detach().max(dim=1)
                y_list.append(y_root.reshape(-1, 1))
            pred_list.append(pred_discrete)
        
        y_all = torch.stack(y_list, 1).reshape(-1)
        pred_all = torch.stack(pred_list, 0).reshape(-1)
        acc = torch.sum(pred_all==y_all) / pred_all.shape[0]
        if self.config['task']['target'] == 'shared':
            precision, recall, f1 = self.classification_metric(pred_all-1, y_all-1)
        elif self.config['task']['target'] == 'root':
            precision, recall, f1 = self.classification_metric(pred_all, y_all)
        return acc.item(), precision.item(), recall.item(), f1.item()

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
            wandb.init(project='EDA_benchmark', name = self.config['task']['name']+'_'+str(self.config['task']['type'])+'_'+self.config['model']['name'])
    def save_model(self, valid_metric, epoch_idx):
        if valid_metric < self.best_valid_metric:
            self.best_valid_metric = valid_metric
            delete_file_with_head(self.train_folder, 'model')
            torch.save(self.model.state_dict(), self.train_folder+'model'+'_epoch'+str(epoch_idx)+'.pth')

    def raytune(self, tune_config, num_samples, num_cpu, num_gpu_per_trial):
        # init ray tune
        reporter = CLIReporter(parameter_columns=['hidden_dim'],metric_columns=['loss', 'acc'])
        dropout_p = tune.choice(tune_config['dropout'])
        tune_parameter_config = {
        'name': tune_config['name'],
        'hidden_dim': 32 * hp.randint('hidden_dim', int(tune_config['hidden_dim'][0]), int(tune_config['hidden_dim'][1])),
        'num_layers': hp.randint('num_layers', int(tune_config['num_layers'][0]), int(tune_config['num_layers'][1])),
        'lr': hp.uniform('lr', float(tune_config['lr'][0]), float(tune_config['lr'][1])),
        'dropout': dropout_p,
        'mlp_out': {'num_layer': hp.randint('mlp_out', int(tune_config['mlp_out']['num_layer'][0]), 
                                            int(tune_config['mlp_out']['num_layer'][1]))},
        'node_input_dim': 4,
        'num_layers': hp.randint('num_layers', int(tune_config['num_layers'][0]), int(tune_config['num_layers'][1])),
        'lr': hp.uniform('lr', float(tune_config['lr'][0]), float(tune_config['lr'][1])),
        'dropout': dropout_p,
        'node_input_dim': self.config['model']['node_input_dim'],
        'edge_input_dim': self.config['model']['edge_input_dim'],
        'num_class': 3,
        }
        tune_parameter_config = {**self.config['model'], **tune_parameter_config}
        scheduler = ASHAScheduler(
            max_t=2000,
            grace_period=1000,
            reduction_factor=2)
        
        hyperopt_search = HyperOptSearch(tune_parameter_config, metric='loss', mode='min')
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_ray),
                resources={'cpu': num_cpu, 'gpu': num_gpu_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='loss',
                mode='min',
                scheduler=scheduler,
                num_samples=num_samples,
                search_alg=hyperopt_search,   
            ),
            run_config=RunConfig(progress_reporter=reporter),
        )
        results = tuner.fit()
        
        best_result = results.get_best_result('loss', 'min')

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics['loss']))
        
        print("Best trial final validation acc root: {}".format(
            best_result.metrics['acc']))

        print("Best trial final validation precision shared: {}".format(
            best_result.metrics['precision']))
        
        print("Best trial final validation recall shared: {}".format(
            best_result.metrics['recall']))
        
        print("Best trial final validation f1 shared: {}".format(
            best_result.metrics['f1']))
        
        
        
        '''precision_result = results.get_best_result('precision', 'max')
        print("Best trial config: {}".format(precision_result.config))
        print("Best trial final validation loss: {}".format(
            precision_result.metrics['loss']))

        print("Best trial final validation precision shared: {}".format(
            precision_result.metrics['shared_precision']))
        
        print("Best trial final validation recall shared: {}".format(
            precision_result.metrics['shared_recall']))
        
        print("Best trial final validation f1 shared: {}".format(
            precision_result.metrics['shared_f1']))
        
        print("Best trial final validation precision root: {}".format(
            precision_result.metrics['root_precision']))
        
        print("Best trial final validation recall root: {}".format(
            precision_result.metrics['root_recall']))
        
        print("Best trial final validation f1 root: {}".format(
            precision_result.metrics['root_f1']))'''

    
        
            
                
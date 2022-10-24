from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import yaml
import os
import glob
# from torchsummary import summary
from datetime import datetime
import time
from tree_sitter import Language, Parser
from torch.utils.data import SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_recall_curve

from data.vocab import Vocab
from data.oj_astnn_dataset import *
from data.oj_astnn_loader import make_clone_data_loader
from data.utils import *
from data.lang import Lang

from constants import PAD_IDX

from network.graph_models import HierarchicalGatedModel, HierarchicalHGTModel, HierarchicalHGT2Seq, HierarchicalHGTTransformer
from network.mask_graph_models import MaskedHGTTransformer
from network.model_from_cp import ModelFromCp, TripletModelFromCp
from network.clone_astnn import CloneASTNNModel
from network.utils import Log


from losses import MaskedCrossEntropyLoss, FocalLoss, TripletLosses
from metrics import corpus_bleu

from concurrent.futures import ProcessPoolExecutor
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', type = int, default = 1, help = 'use cuda or not')
    ap.add_argument('--device-index', type = int, default = 2, help = 'index of nvidia gpu in cuda use case')
    ap.add_argument('--apply-scheduler', type = int, default = 1, help = 'use scheduler for learning rate or not')
    ap.add_argument('--apply-clip-gradient', type = int, default = 0, help = 'clip gradient or not')
    ap.add_argument('--save-best-checkpoint', help='The end date of inference.', type = int, default = 0)
    ap.add_argument('--verbose', help='trace logs in console', type = int, default = 1)
    ap.add_argument('--parser-path', type = str, default = 'language/c.so')
    ap.add_argument('--file-ext', type = str, default = 'c', help = 'file extension in dataset')
    ap.add_argument('--language', type = str, default = 'c', help = 'programming language in dataset')
    ap.add_argument('--config-path', type = str, help = 'path of the configuration file')
    ap.add_argument('--pretrain-cfg-path', type = str, default = 'configs/config_mask_cpp.yml')
    ap.add_argument('--apply-early-stopping', type = int, default = 1, help = 'training process early stop or not')
    ap.add_argument('--sample-classes', default = 0, type = int, help = 'sample the number of classes to train/evaluate')
    ap.add_argument('--dgl-format', default = 1, type = int, help = 'dgl or torch-geometric')
    ap.add_argument('--num-node-types', default = 1, type = int, choices = [1, 2], help = 'the number of the node types')
    ap.add_argument('--num-workers', default = 0, type = int, help = 'number of the workers in DataLoader')
    ap.add_argument('--func', default = 'gru', choices = ['linear', 'gru', 'ffd'], type = str)
    ap.add_argument('--test', default = 0, type = int)
    ap.add_argument('--tree-aggr', default = 'attention', choices = ['attention', 'max-pooling'], type = str)
    ap.add_argument('--graph-aggr', default = 'conv-attention', choices = ['conv-attention', 'attention', 'max-pooling'], type = str)
    ap.add_argument('--pos-encoding', default = 1, type = int)
    ap.add_argument('--scratch-train', default = 1, type = int)
    ap.add_argument('--task', type = str, default = 'clone', choices = ['classification', 'summarization'])
    ap.add_argument('--dataset', type = str, default = 'c')
    return ap.parse_args()


def process(data, s, e):
    return [data[i] for i in range(s, e)]

def pairwise_cosine_similarity(h):
    h_norm = torch.nn.functional.normalize(h, dim=1)
    sim = torch.mm(h_norm, h_norm.transpose(0, 1))
    return sim


def compute_pairwise_scores(h, pids):
    sim = pairwise_cosine_similarity(h)
    inds = torch.triu_indices(len(pids), len(pids), offset=1)
    sim = sim[inds[0], inds[1]]
    positive = pids[inds[0]] == pids[inds[1]]
    s_p = sim[positive]
    s_n = sim[~positive]
    return s_p, s_n

class CircleLoss(nn.Module):
    def __init__(self, gamma, m):
        super().__init__()
        self.gamma = gamma
        self.m = m

    def forward(self, s_p, s_n):
        alpha_p = torch.clamp_min(1 + self.m - s_p, 0)
        alpha_n = torch.clamp_min(self.m + s_n, 0)
        delta_p = 1 - self.m
        delta_n = self.m
        logit_p = (-self.gamma) * alpha_p * (s_p - delta_p)
        logit_n = self.gamma * alpha_n * (s_n - delta_n)
        return F.softplus(torch.logsumexp(logit_p, dim=0) + torch.logsumexp(logit_n, dim=0))

def get_param_epoch(epoch):
    if epoch == 0:
        return 1, 8 # 1 : 1
    if epoch == 1:
        return 2, [7, 1] # 1 : 3
    if epoch == 2:
        return 3, [5, 2, 1] # 1 : 7
    if epoch == 3:
        return 4, [3, 2, 2, 1]
    if epoch < 7:
        return 8, 1
    if epoch == 7:
        return 7, [2] + [1] * 6
    if epoch == 8:
        return 6, [3] + [1] * 5
    if epoch == 9:
        return 5, [3, 2, 2] + [1] * 2
    if epoch == 10:
        return 4, [3, 3] + [1] * 2
    if epoch == 11:
        return 3, [4, 3, 1]
    return 2, [4, 4]

class Runner:
    def __init__(self, args):
        with open(args.config_path, 'r') as f:
            config = yaml.load(f, Loader = yaml.FullLoader)
        with open(args.pretrain_cfg_path, 'r') as f:
            pretrain_config = yaml.load(f, Loader = yaml.FullLoader)
        # assert config['batch_size'] == 1
        self.device = torch.device(f'cuda:{args.device_index}' if args.cuda else 'cpu')


        parser = Parser()
        parser.set_language(Language(args.parser_path, args.language))
        
        self.vocab_obj = Vocab(config['vocab']['node_token'], config['vocab']['node_type'])
        config['vocab_token_size'] = self.vocab_obj.vocab_token_size
        config['vocab_type_size'] = self.vocab_obj.vocab_type_size
        if args.test:
            data_path = config['dataset_paths']['data']
            test_path = config['dataset_paths']['test']
            triplet = True
            self.triplet = triplet
            print('-----------------------------------Evaluating-----------------------------------')
            self.test_dataset = OJASTNNDataset(self.vocab_obj, data_path, test_path, parser, args.dgl_format, args.num_node_types)
            self.test_loader = make_clone_data_loader(self.test_dataset, self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)
        elif args.task == 'clone':
            if args.dataset == 'c':
                data_path = config['dataset_paths']['data']
                train_path = config['dataset_paths']['train']
                val_path = config['dataset_paths']['val']
                test_path = config['dataset_paths']['test']
                self.train_dataset =  OJASTNNDataset(self.vocab_obj, data_path, train_path, parser, args.dgl_format, args.num_node_types)
                self.train_loader = make_clone_data_loader(self.train_dataset,self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, shuffle = True, num_workers = args.num_workers)
                self.val_dataset = OJASTNNDataset(self.vocab_obj, data_path, val_path, parser, args.dgl_format, args.num_node_types)
                self.val_loader = make_clone_data_loader(self.val_dataset, self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)
                
                self.test_dataset = OJASTNNDataset(self.vocab_obj, data_path, test_path, parser, args.dgl_format, args.num_node_types)
                self.test_loader = make_clone_data_loader(self.test_dataset, self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)

        print(args.task )
        if args.task == 'clone':
            if args.dgl_format:
                if args.num_node_types == 1:
                    if args.scratch_train == 1:
                        print('FROM SCRATCH ...')
                        all_triplets = [('node', 'ast_edge', 'node'), #000
                                        ('node', 'control_flow_edge', 'node'), #011
                                        ('node', 'next_stmt_edge','node'),
                                        ('node', 'data_flow_edge', 'node')] 
                        metadata = (
                            {'node': 0},
                            dict(zip(all_triplets, range(len(all_triplets))))     
                        )
                        backbone = HierarchicalHGTModel(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding).to(self.device)
                        self.model = CloneASTNNModel(backbone, config['out_channels'], args.dgl_format, args.num_node_types, args.graph_aggr).to(self.device)
                    else:
                        print('FROM PRETRAINING TASK ...')
                        all_triplets = [('node', 'ast_edge', 'node'), #000
                                        ('node', 'control_flow_edge', 'node'), #011
                                        ('node', 'next_stmt_edge','node'),
                                        ('node', 'data_flow_edge', 'node')] 
                        metadata = (
                            {'node': 0},
                            dict(zip(all_triplets, range(len(all_triplets))))     
                        )
                        pre_vocab_obj = Vocab(pretrain_config['vocab']['node_token'], pretrain_config['vocab']['node_type'])
                        vocab_token_size = self.vocab_obj.vocab_token_size
                        vocab_type_size = self.vocab_obj.vocab_type_size
                        pretrain_config['vocab_token_size'] = vocab_token_size
                        pretrain_config['vocab_type_size'] = vocab_type_size
                        lang_obj = Lang(pretrain_config['vocab']['pcn_cpp_1000'])
                        pretrain_config['vocab_comment_size'] = lang_obj.vocab_size
                        pretrained_model = MaskedHGTTransformer(pretrain_config, metadata, lang_obj.word2index, args.dgl_format, args.func, args.num_node_types, tree_aggr = 'max-pooling', pos_encoding = 0, apply_copy = 0).to(self.device)
                        cp = torch.load(pretrain_config['old_checkpoint_path'], map_location = self.device)
                        model = cp['model']
                        token_weight = torch.rand(vocab_token_size + 1, 256)
                        type_weight = torch.rand(vocab_type_size + 1, 256)
                        torch.nn.init.xavier_normal_(token_weight)
                        torch.nn.init.xavier_normal_(type_weight)
                        token_weight[:model['node_embedding_layer.token_embedding.weight'].size(0), :] = model['node_embedding_layer.token_embedding.weight']
                        type_weight[:model['node_embedding_layer.type_embedding.weight'].size(0), :] = model['node_embedding_layer.type_embedding.weight']
                        for k in ['node_embedding_layer.token_embedding.weight', 'ast_embedding_layer.embedding_layer.token_embedding.weight', 'tbcnn_layer.embedding_layer.token_embedding.weight']:
                            model[k] = token_weight
                        for k in ['node_embedding_layer.type_embedding.weight', 'ast_embedding_layer.embedding_layer.type_embedding.weight', 'tbcnn_layer.embedding_layer.type_embedding.weight']:
                            model[k] = type_weight
                        pretrained_model.load_state_dict(model)
                        self.model = CloneASTNNModel(pretrained_model, config['out_channels'], args.dgl_format, args.num_node_types, args.graph_aggr).to(self.device)
                elif args.num_node_types == 2:
                    print('[MODEL] 2 node types HGT')
                    all_triplets = [('ast_node', 'ast_edge', 'ast_node'), #000
                        ('ast_node', 'control_flow_edge', 'stmt_node'), #011
                        ('stmt_node', 'next_stmt_edge', 'stmt_node'), #112
                        ('stmt_node', 'ast_edge', 'stmt_node'), #110
                        ('stmt_node', 'control_flow_edge', 'stmt_node'), #111
                        ('stmt_node', 'ast_edge', 'ast_node'), #100
                        ('stmt_node', 'control_flow_edge', 'ast_node'), #101
                        ('ast_node', 'ast_edge', 'stmt_node'), #010
                        ('ast_node', 'control_flow_edge', 'ast_node'), #001
                    ] 
                    metadata = (
                        {'ast_node': 0, 'stmt_node': 1},
                        dict(zip(all_triplets, range(len(all_triplets))))     
                    ) 
                    self.model = HierarchicalHGTModel(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding).to(self.device)
            else:
                metadata = (
                    ['ast_node', 'stmt_node'],
                    [('ast_node', 'ast_edge', 'ast_node'), #000
                    ('ast_node', 'control_flow_edge', 'stmt_node'), #011
                    ('stmt_node', 'next_stmt_edge', 'stmt_node'), #112
                    ('stmt_node', 'ast_edge', 'stmt_node'), #110
                    ('stmt_node', 'control_flow_edge', 'stmt_node'), #111
                    ('stmt_node', 'ast_edge', 'ast_node'), #100
                    ('stmt_node', 'control_flow_edge', 'ast_node'), #101
                    ('ast_node', 'ast_edge', 'stmt_node'), #010
                    ('ast_node', 'control_flow_edge', 'ast_node'), #001
                    ]) 
                self.model = HierarchicalHGTModel(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding).to(self.device)
            
            self.criterion = nn.BCELoss().to(self.device)
            self.focal_loss = FocalLoss(0.6, 3.2)
            self.get_accuracy = self.get_cross_acc
            

        elif args.task == 'summarization':
            print('[MODEL] DGL 1 node type HGT')
            all_triplets = [('node', 'ast_edge', 'node'), #000
                            ('node', 'control_flow_edge', 'node'), #011
                            ('node', 'next_stmt_edge','node')] 
            metadata = (
                {'node': 0},
                dict(zip(all_triplets, range(len(all_triplets))))     
            ) 
            config['vocab_comment_size'] = self.lang_obj.vocab_size
            self.model = HierarchicalHGT2Seq(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding, sos_index = self.lang_obj.get_word_index('[SOS]')).to(self.device)
            # self.model = HierarchicalHGTTransformer(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, pos_encoding = args.pos_encoding).to(self.device)
            
            self.criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
            # self.criterion = MaskedCrossEntropyLoss().to(self.device)
        if args.test == 0:
            self.optimizer = getattr(torch.optim, config['optimizer']['name'])(self.model.parameters(), **config['optimizer']['params'])
            if args.apply_scheduler:
                print('-----------------------------SCHEDULING---------------------------------')
                if args.verbose:
                    config['scheduler']['params']['verbose'] = True
                if config['scheduler']['name'] == 'OneCycleLR':
                    config['scheduler']['params']['steps_per_epoch'] = len(self.train_loader)
                config['scheduler']['params']['verbose'] = False
                self.scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])(self.optimizer, **config['scheduler']['params'])
            else:
                print('-----------------------------NO SCHEDULING---------------------------------')
                self.scheduler = None
            timestamp = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
            self.train_logs_path = os.path.join(config['log_dir'], f'train_log_{timestamp}.csv')
            self.loss_traces = []
            self.apply_clip_grad = args.apply_clip_gradient
            self.max_threshold_clip_grad = config['clip_gradient']['max_threshold'] if config['clip_gradient'] else None
            self.start_epoch = 0
        

        self.config = config
        self.args = args

        if args.task == 'clone':
            self.train_one_epoch = self.train_clone_one_epoch
            self.evaluate = lambda loader, verbose = True: self.evaluate_clone(loader, verbose)
    def set_device_tensors(self, ast_node_index, tree, graphs, in_degrees, labels, device, is_bucket = False):
        new_ast_node_index = {
            k: {
                'node_type_index': v['node_type_index'].to(device),
                'node_sub_token_ids': v['node_sub_token_ids'].to(device)
            }
            for k, v in ast_node_index.items()
        }
        if is_bucket:
            new_buckets = {}
            for size, batch in tree.items():
                new_buckets[size] = {}
                for k, v in batch.items():
                    new_buckets[size][k] = v.to(device) if k != 'batch_tree_index' else v
        else:
            new_tree = dict({k: v.to(device) if k != 'batch_tree_index' else v for k, v in tree.items()})

        new_graphs = graphs.to(device)
        new_labels = labels.to(device)

        new_in_degrees = {}
        for node_type, value in in_degrees.items():
            new_in_degrees[node_type] = value.to(device)
        if is_bucket:
            return new_ast_node_index, new_buckets, new_graphs, new_in_degrees, new_labels
        return new_ast_node_index, new_tree, new_graphs, new_in_degrees, new_labels
    @torch.no_grad()
    def get_cross_acc(self, logits, labels):
        preds = torch.argmax(logits, dim = 1)
        score = (preds == labels).float()
        return torch.mean(score)
    @torch.no_grad()
    def get_bin_acc(self, logits, labels):
        preds = (logits > 0.5).float()
        score = (preds == labels).float()
        return torch.mean(score)
    def map_at_r(self, sim, pids):
        r = torch.bincount(pids) - 1
        max_r = r.max()

        mask = torch.arange(max_r)[None, :] < r[pids][:, None]

        sim = sim.clone()
        ind = np.diag_indices(len(sim))
        sim[ind[0], ind[1]] = -np.inf

        _, result = torch.topk(sim, max_r, dim=1, sorted=True)

        tp = (pids[result] == pids[:, None])
        tp[~mask] = False

        valid = r[pids] > 0

        p = torch.cumsum(tp, dim=1).float() / torch.arange(1, max_r+1)[None, :]
        ap = (p * tp).sum(dim=1)[valid] / r[pids][valid]

        return ap.mean().item()

    @torch.no_grad()
    def evaluate_clone(self, loader, verbose = True):
        acc_acc = 0
        acc_loss = 0
        count = 0
        print('-----------------------------------Evaluating-----------------------------------')
        self.model.eval()
        bar = tqdm(loader, total = len(loader)) if verbose else loader
        total_lbs = []
        total_preds = []
        eval_loss = 0
        nb_eval_steps = 0
        for num_nodes, ast_node_index, buckets, graphs, in_degrees, labels in bar:
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, labels = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, labels, device = self.device, is_bucket = True)
            self.optimizer.zero_grad()
            outputs = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees, labels)
            loss = self.criterion(outputs, labels)
            preds = (outputs > 0.5).float()
            total_lbs.append(labels.cpu().numpy())
            total_preds.append(preds.cpu().numpy())
            eval_loss += loss.mean().item()
            nb_eval_steps += 1
        total_preds=np.concatenate(total_preds,0)
        total_lbs=np.concatenate(total_lbs,0)
        acc = np.array(total_preds == total_lbs).mean()
        p, r, f1, _ = precision_recall_fscore_support(total_lbs, total_preds, average = 'binary')
        eval_loss = eval_loss / nb_eval_steps
        return eval_loss, acc, p, r, f1
    @torch.no_grad()
    def test_clone(self, verbose = True):
        acc_acc = 0
        acc_loss = 0
        count = 0
        

        self.model.eval()
        loader = self.test_loader
        dataset = self.test_dataset
        bar = tqdm(loader, total = len(loader)) if verbose else loader
        total_outputs = []
        total_lbs = []
        eval_loss = 0
        nb_eval_steps = 0
        for num_nodes, ast_node_index, buckets, graphs, in_degrees, labels in bar:
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, labels = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, labels, device = self.device, is_bucket = True)
            # self.optimizer.zero_grad()
            outputs = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees, labels)
            loss = self.criterion(outputs, labels)
            total_lbs.append(labels.cpu().numpy())
            total_outputs.append(outputs.cpu().numpy())
        total_outputs=np.concatenate(total_outputs,0)
        total_lbs=np.concatenate(total_lbs,0)

        th = 37
        total_preds = (total_outputs > th / 100).astype(int)
        acc = np.mean(total_preds == total_lbs)
        p, r, f1, _ = precision_recall_fscore_support(total_lbs, total_preds, average = 'binary')
        print(f' acc: {acc: .4f} p: {p: .4f} r: {r: .4f} f1: {f1: .4f}')
    def train_clone_one_epoch(self):
        self.model.train()
        if self.args.verbose:
            bar = tqdm(self.train_loader, total = len(self.train_loader))
        else:
            bar = self.train_loader
        acc_loss = 0
        acc_acc = 0
        for i, (num_nodes, ast_node_index, buckets, graphs, in_degrees, labels) in enumerate(bar):
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, labels = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, labels, device = self.device, is_bucket = True)
            self.optimizer.zero_grad()
            outputs = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees, labels)
            loss = self.criterion(outputs, labels) + 3.5 * self.focal_loss(outputs, labels)
            acc = self.get_bin_acc(outputs, labels)
            # bs = embeddings.shape[0] // 3
            # anchor_logits = logits[:bs]
            # anchor_lbs = labels[:bs]
            # anchor_embeds = embeddings[:bs]
            # pos_embeds = embeddings[bs: 2 * bs]
            # neg_embeds = embeddings[-bs:]
            # loss = self.criterion(anchor_logits, anchor_lbs, anchor_embeds, pos_embeds, neg_embeds)
            loss.backward()
            # print(self.model.node_embedding_layer.token_embedding.weight.grad.min(), self.model.node_embedding_layer.token_embedding.weight.grad.max())
            if self.apply_clip_grad:
                if 'value' in self.config['clip_gradient']['name']:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value = self.max_threshold_clip_grad)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_threshold_clip_grad, norm_type=2)

            self.optimizer.step()
            
            if self.args.verbose:
                bar.set_description(f'iter: {i} loss {loss.detach().cpu().item():.5f} acc {acc.cpu().item(): .4f}')
            acc_loss += loss.detach().cpu().item()
            acc_acc += acc.cpu().item()

            if self.scheduler: self.scheduler.step()
        
        return acc_loss / len(self.train_loader), acc_acc / len(self.train_loader)
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        if 'scheduler' in checkpoint and hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
    def save_checkpoint(self, epoch, save_best_checkpoint = False):
        path = os.path.join(self.config['checkpoint_dir'], 'cp_best.tar' if save_best_checkpoint else f'cp_{epoch}.tar')
        saved_checkpoint_data = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler:
            saved_checkpoint_data['scheduler'] = self.scheduler.state_dict()      
        torch.save(saved_checkpoint_data, path)
    def create_folders(self):
        if not os.path.exists(self.config['checkpoint_dir']):
            os.mkdir(self.config['checkpoint_dir'])
        if not os.path.exists(self.config['log_dir']):
            os.mkdir(self.config['log_dir'])
    def train(self):
        Log.write_header(self.train_logs_path, 'Epoch, train_time, train_loss, train_acc, val_time, val_loss, val_acc, val_p, val_r, val_f1, test_time, test_loss, test_acc, test_p, test_r, test_f1')
        # min_loss = float('inf')
        # val_start_time = time.time()
        # val_perplexity, val_map = self.evaluate(self.val_loader)
        # val_end_time = time.time()
        # val_time = val_end_time - val_start_time
        # print(val_perplexity, val_map)
        for epoch in range(self.start_epoch, self.config['epochs']):
            # self.train_sampler.set_value(16, 1)
            if self.args.verbose:
                print(f'Epoch {epoch}')
            train_start_time = time.time()
            train_loss, train_acc = self.train_one_epoch()
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            self.save_checkpoint(epoch)

            val_start_time = time.time()
            val_loss, val_acc, val_p, val_r, val_f1 = self.evaluate(self.val_loader)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time

            test_start_time = time.time()
            test_loss, test_acc, test_p, test_r, test_f1 = self.evaluate(self.test_loader)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time
            if self.args.verbose:
                print(f'[INFO] Epoch: {epoch} train_loss: {train_loss: .3f} train_acc: {train_acc: .3f} val_loss: {val_loss: .3f} val_acc: {val_acc: .3f} test_loss: {test_loss: .3f} test_acc: {test_acc: .3f}')
                # if self.args.task == 'classification':
                #     print(f'[INFO] Epoch: {epoch} train_loss: {train_loss: .3f} train_acc: {train_acc: .3f} val_loss: {val_loss: .3f} val_acc: {val_acc: .3f} test_acc: {test_acc: .3f}')
                # else:
                #     print(f'[INFO] Epoch: {epoch} train_loss: {train_loss: .3f} train_bleu: {train_acc: .3f} val_loss: {val_loss: .3f} val_bleu: {val_acc: .3f} test_bleu: {test_bleu: .3f}')
            
            Log.write_log(self.train_logs_path, f'{epoch}, {train_time}, {train_loss}, {train_acc}, {val_time}, {val_loss}, {val_acc}, {val_p}, {val_r}, {val_f1}, {test_time}, {test_loss}, {test_acc}, {test_p}, {test_r}, {test_f1}')
            
            # if self.args.save_best_checkpoint:
            #     if min_loss > val_loss:
            #         self.save_checkpoint(epoch, True)
            #         min_loss = val_loss
            # else:
            #     if epoch % self.config['checkpoint_per_epoch'] == 0:
            #         self.save_checkpoint(epoch)

            if self.args.apply_early_stopping and self.early_stopping(train_loss):
                break
        if epoch % self.config['checkpoint_per_epoch'] != 0:
            self.save_checkpoint(epoch)
    def early_stopping(self, loss):
        patience = self.config['early_stopping']['patience']
        if len(self.loss_traces) == 0: 
            self.loss_traces.append(loss)
            return False
        if self.loss_traces[0] > loss:
            self.loss_traces = [loss]
            return False
        if len(self.loss_traces) < patience: 
            self.loss_traces.append(loss)
            return False
        return True

    def run(self):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

        torch.autograd.set_detect_anomaly(True)
        if os.path.exists(self.config['old_checkpoint_path']):
            print('LOADDDDDD')
            self.load_checkpoint(self.config['old_checkpoint_path'])
        self.test_clone()
        if self.args.test:
            self.test_clone()
            # test_loss, test_acc = self.evaluate(self.test_loader)
            # print(f'Test loss: {test_loss: .4f} Test accuracy: {test_acc: .4f}')
        else:
            # test_loss, test_acc = self.evaluate(self.test_loader)
            # val_loss, val_acc = self.evaluate(self.val_loader)
            # print(f'Val loss: {val_acc: .4f} Test accuracy: {test_acc: .4f}')
            self.create_folders()
            self.train()
        


if __name__ == '__main__':
    Runner(parse_args()).run()

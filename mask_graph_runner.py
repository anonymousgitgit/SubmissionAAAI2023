from collections import defaultdict
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import yaml
import os
import glob

from datetime import datetime
import time
from tree_sitter import Language, Parser
from tokenizers import Tokenizer, decoders
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Sequence, Strip, Lowercase, NFD, StripAccents

from torch.profiler import profile, record_function, ProfilerActivity

from data.vocab import Vocab
from data.pcn_dataset import MaskedPCNDataset
from data.slm_dataset import MaskedSLMDataset
from data.csn_dataset import MaskedCSNDataset


from data.utils import *
from data.lang import Lang

from constants import PAD_IDX
from network.model_from_cp import ModelFromCp

from network.graph_models import HierarchicalGatedModel, HierarchicalHGTModel, HierarchicalHGT2Seq, HierarchicalHGTTransformer
from network.mask_graph_models import MaskedHGTTransformer
from network.utils import Log

from losses import CopyGeneratorCriterion

from modules.utils import tens2sen

from metrics import corpus_bleu

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', type = int, default = 0, help = 'use cuda or not')
    ap.add_argument('--device-index', type = int, default = 0, help = 'index of nvidia gpu in cuda use case')
    ap.add_argument('--apply-scheduler', type = int, default = 0, help = 'use scheduler for learning rate or not')
    ap.add_argument('--apply-clip-gradient', type = int, default = 0, help = 'clip gradient or not')
    ap.add_argument('--save-best-checkpoint', help='The end date of inference.', type = int, default = 0)
    ap.add_argument('--verbose', help='trace logs in console', type = int, default = 1)
    ap.add_argument('--parser-path', type = str, default = 'language/java.so')
    ap.add_argument('--file-ext', type = str, default = 'java', help = 'file extension in dataset')
    ap.add_argument('--language', type = str, default = 'java', help = 'programming language in dataset')
    ap.add_argument('--config-path', type = str, default = 'configs/config_mask_java_dgl_csn.yml', help = 'path of the configuration file')
    ap.add_argument('--apply-early-stopping', type = int, default = 1, help = 'training process early stop or not')
    ap.add_argument('--sample-classes', default = 0, type = int, help = 'sample the number of classes to train/evaluate')
    ap.add_argument('--dgl-format', default = 1, type = int, help = 'dgl or torch-geometric')
    ap.add_argument('--num-node-types', default = 1, type = int, choices = [1, 2], help = 'the number of the node types')
    ap.add_argument('--num-workers', default = 8, type = int, help = 'number of the workers in DataLoader')
    ap.add_argument('--func', default = 'ffd', choices = ['linear', 'gru', 'ffd'], type = str)
    ap.add_argument('--test', default = 0, type = int)
    ap.add_argument('--copy-attn', default = 0, type = int)
    ap.add_argument('--tree-aggr', default = 'max-pooling', choices = ['attention', 'max-pooling'], type = str)
    ap.add_argument('--graph-aggr', default = 'max-pooling', choices = ['conv-attention', 'attention', 'max-pooling'], type = str)
    ap.add_argument('--pos-encoding', default = 0, type = int)
    ap.add_argument('--task', type = str, default = 'summarization', choices = ['classification', 'summarization'])
    ap.add_argument('--dataset', type = str, default = 'java')
    ap.add_argument('--apply-copy', default = 0, type = int)
    ap.add_argument('--has-data-flow', type = int, default = 1)
    ap.add_argument('--has-cdg', type = int, default = 1)
    ap.add_argument('--pretrain-config-path', type = str)
    ap.add_argument('--fine-tune', type = int, default = 0)
    ap.add_argument('--extract-features', type = int, default = 0)

    return ap.parse_args()



class Runner:
    def __init__(self, args):
        with open(args.config_path, 'r') as f:
            config = yaml.load(f, Loader = yaml.FullLoader)
        # assert config['batch_size'] == 1
        self.device = torch.device(f'cuda:{args.device_index}' if args.cuda else 'cpu')

        parser = Parser()
        parser.set_language(Language(args.parser_path, args.language))
        
        self.vocab_obj = Vocab(config['vocab']['node_token'], config['vocab']['node_type'])
        config['vocab_token_size'] = self.vocab_obj.vocab_token_size
        config['vocab_type_size'] = self.vocab_obj.vocab_type_size
        self.cum_steps = 0
        if args.test:
            self.lang_obj = Lang(config['vocab']['pcn_cpp_1000'])

            train_paths = glob.glob(os.path.join(config['dataset_paths']['train'], '*', f'*.{args.file_ext}'))
            val_paths = glob.glob(os.path.join(config['dataset_paths']['val'], '*', f'*.{args.file_ext}'))
            test_paths = glob.glob(os.path.join(config['dataset_paths']['test'], '*', f'*.{args.file_ext}'))
            
            train_texts = list(map(read_file, train_paths))
            val_texts = list(map(read_file, val_paths))
            test_texts = list(map(read_file, test_paths))

            max_seq_length = config['max_seq_length']
            self.train_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, train_paths, train_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)
            self.valid_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, val_paths, val_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)
            self.test_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, test_paths, test_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)

            self.train_loader = make_data_loader(self.train_dataset, self.vocab_obj, batch_size = config['batch_size'], task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, shuffle = True, num_workers = args.num_workers)
            self.valid_loader = make_data_loader(self.valid_dataset, self.vocab_obj, batch_size = config['batch_size'],  task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)
            self.test_loader = make_data_loader(self.test_dataset, self.vocab_obj, batch_size = config['batch_size'],  task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)

        elif args.task == 'classification':
            if args.dataset == 'c':
                train_paths = glob.glob(os.path.join(config['dataset_paths']['train'], '*', f'*.{args.file_ext}'))
                val_paths = glob.glob(os.path.join(config['dataset_paths']['val'], '*', f'*.{args.file_ext}'))
                test_paths = glob.glob(os.path.join(config['dataset_paths']['test'], '*', f'*.{args.file_ext}'))    

                train_texts = list(map(read_file, train_paths))
                val_texts = list(map(read_file, val_paths))
                test_texts = list(map(read_file, test_paths))

                self.train_dataset =  OJDataset(self.vocab_obj, train_paths, train_texts, parser, args.dgl_format, args.num_node_types)
                self.train_loader = make_data_loader(self.train_dataset,self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, shuffle = True, num_workers = args.num_workers)
                self.valid_dataset = OJDataset(self.vocab_obj, val_paths, val_texts, parser, args.dgl_format, args.num_node_types)
                self.valid_loader = make_data_loader(self.valid_dataset, self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)
                
                self.test_dataset = OJDataset(self.vocab_obj, test_paths, test_texts, parser, args.dgl_format, args.num_node_types)
                self.test_loader = make_data_loader(self.test_dataset, self.vocab_obj, batch_size = config['batch_size'], num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers)
           
        elif args.task == 'summarization':
            if args.dataset == 'cpp':
                from data.pcn_loader import make_data_loader

                self.lang_obj = Lang(config['vocab']['pcn_cpp_1000'])

                train_paths = glob.glob(os.path.join(config['dataset_paths']['train'], '*', f'*.{args.file_ext}'))
                val_paths = glob.glob(os.path.join(config['dataset_paths']['val'], '*', f'*.{args.file_ext}'))
                test_paths = glob.glob(os.path.join(config['dataset_paths']['test'], '*', f'*.{args.file_ext}'))
                
                train_texts = list(map(read_file, train_paths))
                val_texts = list(map(read_file, val_paths))
                test_texts = list(map(read_file, test_paths))

                max_seq_length = config['max_seq_length']
                self.train_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, train_paths, train_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)
                self.valid_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, val_paths, val_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)
                self.test_dataset = MaskedPCNDataset(self.vocab_obj, self.lang_obj, test_paths, test_texts, parser, args.dgl_format, args.num_node_types, max_seq_length)
            
            elif args.dataset == 'java':
                from data.slm_loader import make_data_loader
               
                print(f'DFG: {args.has_data_flow} CDG: {args.has_cdg}')
                self.lang_obj = Lang(config['vocab']['slm_java'])
                train_path = config['dataset_paths']['train']
                test_path = config['dataset_paths']['test']
                val_path = config['dataset_paths']['val']

                max_seq_length = config['max_seq_length']
                tokenizer = Tokenizer.from_file(config["vocab"]["tokenizer"])
                tokenizer.decoder = decoders.BPEDecoder()


                self.tokenizer = tokenizer
                self.train_dataset = MaskedSLMDataset(self.vocab_obj, self.lang_obj, train_path, parser, args.dgl_format, args.num_node_types, max_seq_length, tokenizer)
                self.valid_dataset = MaskedSLMDataset(self.vocab_obj, self.lang_obj, val_path, parser, args.dgl_format, args.num_node_types, max_seq_length, tokenizer)
                self.test_dataset = MaskedSLMDataset(self.vocab_obj, self.lang_obj, test_path, parser, args.dgl_format, args.num_node_types, max_seq_length, tokenizer)
            else:
                raise RuntimeError("This dataset type isn't supported")
            print('NUM WORKERS', args.num_workers)
            self.train_loader = make_data_loader(self.train_dataset, self.vocab_obj, batch_size = config['batch_size'], task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, shuffle = True, num_workers = args.num_workers, tokenizer = tokenizer)
            self.valid_loader = make_data_loader(self.valid_dataset, self.vocab_obj, batch_size = config['batch_size'],  task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers, tokenizer = tokenizer)
            self.test_loader = make_data_loader(self.test_dataset, self.vocab_obj, batch_size = config['batch_size'],  task = 'summarization', num_node_types = args.num_node_types, dgl_format = args.dgl_format, training = False, num_workers = args.num_workers, tokenizer = tokenizer)
            # for _ in tqdm(self.train_loader, total = len(self.train_loader)): pass
            if args.apply_copy:

                self.criterion = CopyGeneratorCriterion(self.lang_obj.vocab_size, force_copy = False)
            else:

                self.criterion = nn.CrossEntropyLoss(ignore_index = 0).to(self.device)

        if args.task == 'classification':
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            if args.dgl_format:
                if args.fine_tune == 1:
                    print('OLD CP PATH', args.pretrain_config_path)
                    with open(args.pretrain_config_path, 'r') as f:
                        pre_config = yaml.load(f, Loader = yaml.FullLoader)
                    pre_vocab_obj = Vocab(pre_config['vocab']['node_token'], pre_config['vocab']['node_type'])
                    vocab_obj = Vocab(config['vocab']['node_token'], config['vocab']['node_type'])
                    vocab_token_size = max(pre_vocab_obj.vocab_token_size, vocab_obj.vocab_token_size)
                    vocab_type_size = max(pre_vocab_obj.vocab_type_size, vocab_obj.vocab_type_size)
                    pre_config['vocab_token_size'] = vocab_token_size
                    pre_config['vocab_type_size'] = vocab_type_size
                    lang_obj = Lang(pre_config['vocab']['slm_java'])
                    pre_config['vocab_comment_size'] = lang_obj.vocab_size

                    all_triplets = [('node', 'ast_edge', 'node'), #000
                                            ('node', 'control_flow_edge', 'node'), #011
                                            ('node', 'next_stmt_edge','node'),
                                            ('node', 'data_flow_edge','node')] 
                    metadata = (
                        {'node': 0},
                        dict(zip(all_triplets, range(len(all_triplets))))     
                    ) 
                    pretrained_model = MaskedHGTTransformer(pre_config, metadata, lang_obj.word2index, 1, 'gru', 1, tree_aggr = 'max-pooling', pos_encoding = 0, apply_copy = 0)

                    cp = torch.load(pre_config['old_checkpoint_path'], map_location = self.device)
                    model = cp['model']
                    dim = config['in_channels'] // 2
                    token_weight = torch.rand(vocab_token_size + 1, dim)
                    type_weight = torch.rand(vocab_type_size + 1, dim)
                    torch.nn.init.xavier_normal_(token_weight)
                    torch.nn.init.xavier_normal_(type_weight)
                    token_weight[:model['node_embedding_layer.token_embedding.weight'].size(0), :] = model['node_embedding_layer.token_embedding.weight']
                    type_weight[:model['node_embedding_layer.type_embedding.weight'].size(0), :] = model['node_embedding_layer.type_embedding.weight']
                    for k in ['node_embedding_layer.token_embedding.weight', 'ast_embedding_layer.embedding_layer.token_embedding.weight', 'tbcnn_layer.embedding_layer.token_embedding.weight']:
                        model[k] = token_weight
                    for k in ['node_embedding_layer.type_embedding.weight', 'ast_embedding_layer.embedding_layer.type_embedding.weight', 'tbcnn_layer.embedding_layer.type_embedding.weight']:
                        model[k] = type_weight

                    pretrained_model.load_state_dict(model)
                    self.model = ModelFromCp(pretrained_model, config, 1, 1, graph_aggr = 'max-pooling').to(self.device)
                    
                else:
                    if args.num_node_types == 1:
                        if args.func == 'linear':
                            print('[MODEL] GGNN 1 node type')
                            self.model = HierarchicalGatedModel(config, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding).to(self.device)
                        elif args.func == 'gru':
                            print('[MODEL] DGL 1 node type HGT')
                            all_triplets = [('node', 'ast_edge', 'node'), #000
                                            ('node', 'control_flow_edge', 'node'), #011
                                            ('node', 'next_stmt_edge','node'),
                                            ('node', 'data_flow_edge','node')] 
                            metadata = (
                                {'node': 0},
                                dict(zip(all_triplets, range(len(all_triplets))))     
                            ) 
                            self.model = HierarchicalHGTModel(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding).to(self.device)
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
            
            
            

        elif args.task == 'summarization':
            print('[MODEL] DGL 1 node type HGT')
            all_triplets = [('node', 'ast_edge', 'node'), #000
                            ('node', 'control_flow_edge', 'node'), #011
                            ('node', 'next_stmt_edge','node'),
                            ('node', 'data_flow_edge', 'node')] 
            metadata = (
                {'node': 0},
                dict(zip(all_triplets, range(len(all_triplets))))     
            ) 
            config['vocab_comment_size'] = self.lang_obj.vocab_size
            # self.model = HierarchicalHGT2Seq(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, graph_aggr = args.graph_aggr, pos_encoding = args.pos_encoding, sos_index = self.lang_obj.get_word_index('[SOS]')).to(self.device)
            # self.model = HierarchicalHGTTransformer(config, metadata, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, pos_encoding = args.pos_encoding).to(self.device)
            
            self.model = MaskedHGTTransformer(config, metadata, self.lang_obj.word2index, args.dgl_format, args.func, args.num_node_types, tree_aggr = args.tree_aggr, pos_encoding = args.pos_encoding, apply_copy = args.apply_copy, lang = args.language).to(self.device)

            # self.criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
            # self.criterion = MaskedCrossEntropyLoss().to(self.device)
        if args.test == 0:
            self.optimizer = getattr(torch.optim, config['optimizer']['name'])(self.model.parameters(), **config['optimizer']['params'])
            print("config['scheduler']", config['scheduler'])
            if args.apply_scheduler:
                print('USE SCHEDULER', config['scheduler']['name'])
                if args.verbose:
                    config['scheduler']['params']['verbose'] = True
                config['scheduler']['params']['steps_per_epoch'] = 1#len(self.train_loader)
                if config['scheduler']['name'] == 'LambdaLR':
                    def lr_schedule(epoch):
                        if epoch < 9:
                            return 1.2375 * 1e-5 * epoch + 1e-6
                        else:
                            return 1e-4
                    config['scheduler']['params']['lr_lambda'] = lr_schedule
                self.scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['name'])(self.optimizer, **config['scheduler']['params'])
            else:
                print('NOT USE SCHEDULER')
                self.scheduler = None
            timestamp = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
            self.train_logs_path = os.path.join(config['log_dir'], f'train_log_{timestamp}.csv')
            self.loss_traces = []
            self.apply_clip_grad = args.apply_clip_gradient
            self.max_threshold_clip_grad = config['clip_gradient']['max_threshold'] if config['clip_gradient'] else None
            self.start_epoch = 0
        

        self.config = config
        self.args = args

        if args.task == 'classification':
            self.train_one_epoch = self.train_classification_one_epoch
            self.get_accuracy = self.get_cross_acc
            self.evaluate = lambda loader, verbose = True: self.evaluate_classification(loader, verbose)
        else:
            self.train_one_epoch = self.train_summarization_one_epoch
            self.evaluate = lambda loader, verbose = True: self.evaluate_summarization(loader, verbose)
    
    def set_device_tensors(self, ast_node_index, tree, graphs, in_degrees, extra_src_info, labels, device, is_bucket = False):
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
        if extra_src_info is not None:
            for k, v in extra_src_info.items():
                if k == 'src_vocabs': continue
                extra_src_info[k] = v.to(device)
        if is_bucket:
            return new_ast_node_index, new_buckets, new_graphs, new_in_degrees, extra_src_info, new_labels
        return new_ast_node_index, new_tree, new_graphs, new_in_degrees, extra_src_info, new_labels
    @torch.no_grad()
    def get_cross_acc(self, logits, labels):
        preds = torch.argmax(logits, dim = 1)
        score = (preds == labels).float()
        return torch.mean(score)
    @torch.no_grad()
    def get_bin_acc(self, logits, labels):
        preds = (torch.sigmoid(logits) > 0.5).float()
        score = (preds == labels).float()
        return torch.mean(score)
    @torch.no_grad()
    def evaluate_classification(self, loader, verbose = True):
        acc_acc = 0
        acc_loss = 0
        count = 0
        print('-----------------------------------Evaluating-----------------------------------')
        self.model.eval()
        bar = tqdm(loader, total = len(loader)) if verbose else loader
        
        for num_nodes, ast_node_index, buckets, graphs, in_degrees, labels in bar:
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, _, labels = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, None, labels, device = self.device, is_bucket = True)
            embeddings, logits = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees)
            if len(logits.shape) != len(labels.shape):
                logits = logits.squeeze(1)
            if self.args.dataset == 'devign':
                probs = torch.sigmoid(logits)
                labels = labels.float()
                loss = self.criterion(probs, labels) + 0.5 * self.focal_loss(probs, labels)
            else:
                loss = self.criterion(logits, labels) 
            acc_loss += loss.cpu().item() * len(labels)
            acc_acc += self.get_accuracy(logits, labels).cpu().item() * len(labels)
            count += len(labels)
        return acc_loss / count, acc_acc / count
    @torch.no_grad()
    def evaluate_summarization(self, loader, verbose = True):
        acc_acc = 0
        acc_loss = 0
        count = 0
        print('-----------------------------------Evaluating-----------------------------------')
        self.model.eval()
        bar = tqdm(loader, total = len(loader)) if verbose else loader
        total_preds = []
        total_targets = []
        total_acc = []
        for i, (num_nodes, ast_node_index, buckets, graphs, in_degrees, stmt_ids, extra_src_info, tgt_token_ids, tgt_texts) in enumerate(bar):
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, extra_src_info, tgt_token_ids = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, extra_src_info, tgt_token_ids, device = self.device, is_bucket = True)   
            # tgt_input = label_word_ids[:, :-1]
            # tgt_label = label_word_ids[:, 1:]
            outputs = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees, stmt_ids, extra_src_info, tgt_token_ids,
                                tgt_dict = self.lang_obj.word2index)

            predictions, raw_words = self.predict_one_batch(outputs, self.lang_obj.word2index, extra_src_info['src_vocabs'])

            if hasattr(self, 'tokenizer'):
                preds = []
                for words in raw_words:
                    pred = [self.tokenizer.token_to_id(word) for word in words]
                    preds.append([self.tokenizer.decode(pred)])
                predictions = preds
            # predictions = preds
            _predictions, _tgt_texts = [], []
            for pred, tgt in zip(predictions, tgt_texts):
                pred = pred[0].strip().lower()
                tgt = tgt[0].strip().lower()
                total_acc.append(int(re.sub(r'\s', '', pred) == re.sub(r'\s', '', tgt)))
                _predictions.append([pred])
                _tgt_texts.append([tgt])
            # if outputs.shape[-1] != label_word_ids.shape[1]:
            #     label_word_ids = F.pad(label_word_ids, (0, outputs.shape[-1] - label_word_ids.shape[1]))
            total_preds.extend(_predictions)
            total_targets.extend(_tgt_texts)


        predictions_dict = dict(zip(range(len(total_preds)), total_preds))
        label_word_dict = dict(zip(range(len(total_targets)), total_targets))
        try:
            bleu_score = corpus_bleu(predictions_dict, label_word_dict)[1]
        except:
            bleu_score = 0
        return np.mean(total_acc), bleu_score

    def get_sentence_predictions(self, outputs):
        return list(map(self.lang_obj.get_sentence_from_ids, outputs))
    def train_summarization_one_epoch(self, cum_steps = 0):
        self.model.train()
        if self.args.verbose:
            bar = tqdm(self.train_loader, total = len(self.train_loader))
        else:
            bar = self.train_loader
        
        acc_loss = 0
        acc_bleu = 0

        for i, (num_nodes, ast_node_index, buckets, graphs, in_degrees, stmt_ids, extra_src_info, tgt_token_ids, tgt_texts) in enumerate(bar):
            if any ((tree['batch_children_index'].shape[0] > 60) or (tree['batch_children_index'].shape[1] > 50) or (tree['batch_children_index'].shape[2] > 40) for tree in buckets.values()):
                continue
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, extra_src_info, tgt_token_ids = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, extra_src_info, tgt_token_ids, device = self.device, is_bucket = True)
            self.optimizer.zero_grad()
            # tgt_input = label_word_ids[:, :-1]
            # tgt_label = label_word_ids[:, 1:]
            outputs = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees, stmt_ids, extra_src_info, tgt_token_ids)
            # print('outputs')
            # print(outputs.argmax(dim = 1))
            # print('----------TGT------')
            # print(tgt_token_ids[:, 1:])
            if self.args.apply_copy:
                loss = self.criterion(outputs, extra_src_info['alignments'][:, 1:].contiguous(), tgt_token_ids[:, 1:].contiguous()) 
            else:
                loss = self.criterion(outputs, tgt_token_ids[:, 1:])
            loss.backward()
            if self.apply_clip_grad:
                if 'value' in self.config['clip_gradient']['name']:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value = self.max_threshold_clip_grad)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_threshold_clip_grad, norm_type=2)

            self.optimizer.step()
            # preds = torch.argmax(outputs.detach(), dim = 1)
            # predictions = self.get_sentence_predictions(preds.cpu().numpy().tolist())
            
            # predictions_dict = dict(zip(range(len(predictions)), predictions))
            # label_comment_dict = dict(zip(range(len(label_comments)), label_comments))
            # try:
            #     bleu_score = corpus_bleu(predictions_dict, label_comment_dict)[0]
            # except:
            #     bleu_score = -1
            bleu_score = 0
            if self.args.verbose:
                bar.set_description(f'iter: {i} loss {loss.detach().cpu().item():.5f} bleu: {bleu_score: .4f}')
            # run.log({'epoch': self.current_epoch, 'iter': i, 'loss': loss.detach().cpu().item()})
            acc_loss += loss.detach().cpu().item()

            # acc_bleu += bleu_score

        if self.scheduler: self.scheduler.step()
        
        return acc_loss / len(self.train_loader), 0
    def train_classification_one_epoch(self, cum_steps):
        self.model.train()
        if self.args.verbose:
            bar = tqdm(self.train_loader, total = len(self.train_loader))
        else:
            bar = self.train_loader
        acc_loss = 0
        acc_acc = 0
        for i, (num_nodes, ast_node_index, buckets, graphs, in_degrees, labels) in enumerate(bar):
            if self.args.cuda:
                ast_node_index, buckets, graphs, in_degrees, _, labels = self.set_device_tensors(ast_node_index, buckets, graphs, in_degrees, None, labels, device = self.device, is_bucket = True)
            self.optimizer.zero_grad()
            embeddings, logits = self.model(num_nodes, ast_node_index, buckets, graphs, in_degrees)
            if len(logits.shape) != len(labels.shape):
                logits = logits.squeeze(1)
            if self.args.dataset == 'devign':
                probs = torch.sigmoid(logits)
                labels = labels.float()
                loss = self.criterion(probs, labels) + 0.5 * self.focal_loss(probs, labels)
            else:
                loss = self.criterion(logits, labels) 
            loss.backward()
            # print(self.model.node_embedding_layer.token_embedding.weight.grad.min(), self.model.node_embedding_layer.token_embedding.weight.grad.max())
            if self.apply_clip_grad:
                if 'value' in self.config['clip_gradient']['name']:
                    nn.utils.clip_grad_value_(self.model.parameters(), clip_value = self.max_threshold_clip_grad)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.max_threshold_clip_grad, norm_type=2)

            self.optimizer.step()
            
            acc = self.get_accuracy(logits, labels).cpu().item()
            if self.args.verbose:
                bar.set_description(f'iter: {i} loss {loss.detach().cpu().item():.5f} accuracy: {acc: .4f}')
            acc_loss += loss.detach().cpu().item()
            acc_acc += acc

        if self.scheduler: self.scheduler.step()
        
        return (acc_loss / len(self.train_loader), acc_acc / len(self.train_loader))
  

    def load_checkpoint(self, path):
        print('LOAD CHECKPOINT SUCCESSFULLY!!!')
        checkpoint = torch.load(path, map_location = self.device)
        # pos_weight = torch.rand(100, self.config['in_channels'])
        # max_len = 5000
        # dim = self.config['in_channels']
        # pe = torch.zeros(max_len, dim)
        # position = torch.arange(0, max_len).unsqueeze(1)
        # div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
        #                       -(math.log(10000.0) / dim))).unsqueeze(0)
        # pe[:, 0::2] = torch.sin(position.float() * div_term)
        # pe[:, 1::2] = torch.cos(position.float() * div_term)
        # pe = pe.unsqueeze(0)
        # torch.nn.init.xavier_normal_(pos_weight)
        # checkpoint['model']['tgt_embedder.pos_embedding.weight'] = pos_weight.to(self.device)
        # checkpoint['model']['tgt_embedder.pos_enc_layer.pe'] = pe.to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        if 'steps' in checkpoint:
            self.cum_steps = checkpoint['steps']
            print("STEPS OF CP", self.cum_steps)
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint and hasattr(self, 'scheduler') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
    def save_checkpoint(self, *, epoch = -1, steps = -1, save_best_checkpoint = False):
        path = os.path.join(self.config['checkpoint_dir'], 'cp_best.tar' if save_best_checkpoint else f'cp_{"epoch_" + str(epoch) if epoch != -1 else "steps_" + str(steps)}.tar')
        saved_checkpoint_data = {
            'epoch': epoch,
            'steps': steps, 
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
        if self.args.task == 'classification':
            Log.write_header(self.train_logs_path, 'Epoch, train_time, train_loss, train_acc, val_time, val_loss, val_acc, test_time, test_loss, test_acc')
        else:
            Log.write_header(self.train_logs_path, 'Epoch, train_time, train_loss, train_bleu, val_time, val_acc, val_bleu, test_time, test_acc, test_bleu')
        min_loss = float('inf')
        
        # val_start_time = time.time()
        # val_acc, val_bleu = self.evaluate(self.val_loader)
        # val_end_time = time.time()
        # test_start_time = time.time()
        test_acc, test_bleu = self.evaluate(self.test_loader)
        print(test_acc, test_bleu)
        # test_end_time = time.time()
        # test_time = test_end_time - test_start_time
        # print(f'[INFO] val_acc: {val_acc: .3f} val_bleu: {val_bleu: .3f} test_acc: {test_acc: .3f} test_bleu: {test_bleu: .3f}')
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            if self.args.verbose:
                print(f'Epoch {epoch}')
            self.current_epoch = epoch
            train_start_time = time.time()
            train_loss, train_acc = self.train_one_epoch(self.cum_steps)
            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            self.save_checkpoint(epoch = epoch)

            val_start_time = time.time()
            val_loss, val_acc = self.evaluate(self.valid_loader)
            val_end_time = time.time()
            val_time = val_end_time - val_start_time

            # if self.args.save_best_checkpoint:
            #     if min_loss > val_loss:
            #         self.save_checkpoint(epoch, True)
            #         min_loss = val_loss
            # else:
            #     if epoch % self.config['checkpoint_per_epoch'] == 0:
            #         self.save_checkpoint(epoch)

            test_start_time = time.time()
            test_loss, test_acc = self.evaluate(self.test_loader)
            test_end_time = time.time()
            test_time = test_end_time - test_start_time
            if self.args.verbose:
                if self.args.task == 'classification':
                    print(f'[INFO] Epoch: {epoch} train_loss: {train_loss: .3f} train_acc: {train_acc: .3f} val_loss: {val_loss: .3f} val_acc: {val_acc: .3f} test_acc: {test_acc: .3f}')
                else:
                    print(f'[INFO] Epoch: {epoch} train_loss: {train_loss: .3f} train_bleu: {train_acc: .3f} val_loss: {val_loss: .3f} val_bleu: {val_acc: .3f} test_loss: {test_acc: .3f} test_bleu: {test_acc: .3f}')
            
            Log.write_log(self.train_logs_path, f'{epoch}, {train_time}, {train_loss}, {train_acc}, {val_time}, {val_loss}, {val_acc}, {test_time}, {test_loss}, {test_acc}')
            
            self.cum_steps += len(self.train_loader)
            if self.args.apply_early_stopping and self.early_stopping(train_loss):
                break
        
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
    def predict_one_batch(self, decoder_out, tgt_dict, src_vocabs):
        predictions, raw_words = tens2sen(decoder_out['predictions'], tgt_dict, src_vocabs)
        return predictions, raw_words


    def run(self):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

        torch.autograd.set_detect_anomaly(True)
        
        if self.args.extract_features:
            print('EXTRACT FEATURES')
            self.extract_features(self.train_loader)
        if os.path.exists(self.config['old_checkpoint_path']):
            self.load_checkpoint(self.config['old_checkpoint_path'])
        if self.args.test:
            test_loss, test_acc = self.evaluate(self.test_loader)
            print(f'Test loss: {test_loss: .4f} Test accuracy: {test_acc: .4f}')
        else:
            self.create_folders()
            self.train()
            # run.watch(self.model, log = 'all')
            # with run:
                # self.train()
        


if __name__ == '__main__':
    # with torch.autograd.detect_anomaly():
    Runner(parse_args()).run()
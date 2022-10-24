import torch
import torch.nn as nn
import random
import numpy as np

from network.graph_layers import *

class CloneDetectionModel(nn.Module):   
    def __init__(self, pretrained_model, out_channels, dgl_format, num_node_types, graph_aggr):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.projector = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            # nn.ReLU(inplace = True),
            # nn.Dropout(0.15),
            # nn.Linear(256, 256)
            # nn.Identity()
        )
        self.attn_graph_layer = AttentionGraphLevel(out_channels, graph_aggr)
        self.dgl_format = dgl_format
        self.num_node_types = num_node_types
        self.temperature = 0.5
        self.tau_plus = 0.1 
        self.debiased = True
    
        
    def forward(self, num_nodes, ast_node_index, buckets, graphs, in_degrees, labels): 
        ast_node_index, ast_node_embeddings = self.pretrained_model.ast_embedding_layer(ast_node_index)
        # print('ast node', np.array(ast_node_index).shape, np.unique(ast_node_index).shape)
        sizes = list(buckets.keys())
        if self.training:
            random.shuffle(sizes)
        node_embeddings, batch_tree_index = [], []
        for size in sizes:
            batch = buckets[size]
            each_bucket_embeddings = self.pretrained_model.tbcnn_layer(batch['batch_node_index'], batch['batch_node_type_id'], batch['batch_node_height'], batch['batch_node_sub_tokens_id'], batch['batch_children_index'])
            node_embeddings.append(each_bucket_embeddings)
            batch_tree_index.extend(batch['batch_tree_index'])
        # print('both:', graphs.num_nodes(), len(ast_node_index) + len(batch_tree_index), len(set(ast_node_index) & set(batch_tree_index)), len(set(ast_node_index) | set(batch_tree_index)))
        # print('stmt node', np.array(batch_tree_index).shape, np.unique(batch_tree_index).shape)
        stmt_node_embeddings = torch.cat(node_embeddings, axis = 0)
        if self.dgl_format:
            if self.num_node_types == 1:
                embeddings = torch.cat((ast_node_embeddings, stmt_node_embeddings), dim = 0)
                # ensure that after merging, the order of nodes matches the order in graph
                order_indices = np.argsort(ast_node_index + batch_tree_index)
                # print('a', order_indices.shape)
                node_embeddings = embeddings[order_indices]
                data = {
                    'node': node_embeddings
                }
            else:
                ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
                stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
                data = {
                    'ast_node': ast_order_node_embeddings,
                    'stmt_node': stmt_order_node_embeddings
                }
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(data, graphs = graphs)
        else:
            ast_order_node_embeddings = ast_node_embeddings[np.argsort(ast_node_index)]
            stmt_order_node_embeddings = stmt_node_embeddings[np.argsort(batch_tree_index)]
            graphs['ast_node'].x = ast_order_node_embeddings
            graphs['stmt_node'].x = stmt_order_node_embeddings
            # print(graphs.x_dict)
            all_node_embeddings = self.pretrained_model.hgt_graph_layer(graphs.x_dict, edge_index_dict = graphs.edge_index_dict)

        graph_embeddings = self.attn_graph_layer(all_node_embeddings['node'], num_nodes['num_nodes'], all_node_embeddings['node'][num_nodes['last_stmts']]) 
        # embeddings = self.projector(graph_embeddings[:, :256])
        embeddings = self.projector(graph_embeddings)
        # return embeddings
        batch_size = embeddings.shape[0] // 3
        
        outputs=embeddings.split(batch_size, 0)
        
        prob_1=(outputs[0]*outputs[1]).sum(-1) / self.temperature
        prob_2=(outputs[0]*outputs[2]).sum(-1) / self.temperature
        temp=torch.cat((outputs[0],outputs[1]),0)
        temp_labels=torch.cat((labels,labels),0)
        prob_3= torch.mm(outputs[0],temp.t()) / self.temperature
        mask=labels[:,None]==temp_labels[None,:]
        prob_3=prob_3*(1-mask.float())-1e9*mask.float()
        
        # prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1)
        # loss=torch.log(prob[:,0]+1e-10)
        # loss=-loss.mean()
        # return loss,outputs[0]
        
        # prob=torch.softmax(torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1),-1)
        prob = torch.cat((prob_1[:,None],prob_2[:,None],prob_3),-1)
        prob = prob - prob.max(dim = 1)[0].unsqueeze(1)
        prob = torch.exp(prob)
        pos = prob[:, 0: 1]
        neg = prob[:, 1:]

        
        if self.debiased:
            N = torch.count_nonzero(1 - mask.float(), dim = 1) + 1
            Ng = (-self.tau_plus * N * pos + neg.sum(dim = -1)) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) + 1e-10)).mean()

        # loss=torch.log(prob[:,0]+1e-10)
        # loss=-loss.mean()
        return loss,outputs[0]
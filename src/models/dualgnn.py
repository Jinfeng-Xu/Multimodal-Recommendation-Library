# coding: utf-8
"""
DualGNN: Dual Graph Neural Network for Multimedia Recommendation, IEEE Transactions on Multimedia 2021.
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, softmax
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from utils.graph_cache import GraphCacheManager, DualGNNPreprocessor


class GCN(torch.nn.Module):
    def __init__(self, dataset, batch_size, num_user, num_item, dim_x, aggr_mode, num_layer, has_id, dropout, dim_latent, device, features):
        super(GCN, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_x = dim_x
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.dim_latent = dim_latent
        self.device = device
        self.features = features
        
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_latent)))).to(device)
        self.conv_embed_1 = BaseModel(dim_latent, dim_latent, aggr=aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_1.weight)
        
        self.MLP = nn.Linear(features.size(1), dim_latent)
        nn.init.xavier_normal_(self.MLP.weight)

    def forward(self, edge_index_drop, edge_index, features):
        features_mlp = F.leaky_relu(self.MLP(features))
        
        preference = F.normalize(self.preference)
        features_mlp = F.normalize(features_mlp)
        
        x = torch.cat((preference, features_mlp), dim=0)
        x_hat_1 = self.conv_embed_1(x, edge_index_drop)
        preference = preference + x_hat_1[:self.num_user]
        preference = F.normalize(preference)
        
        x = torch.cat((preference, features_mlp), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1,0]]), dim=1)
        x_hat_1 = self.conv_embed_1(x, edge_index)
        x_hat_1 = F.leaky_relu_(x_hat_1)
        
        return x + x_hat_1, preference


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(BaseModel, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.uniform(self.in_channels, self.weight)

    def forward(self, x, edge_index):
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        # Match MMRec-tmp implementation
        index = user_graph  # [num_users, k]
        u_features = features[index]  # [num_users, k, dim]
        user_matrix = user_matrix.unsqueeze(1)  # [num_users, 1, k]
        u_pre = torch.matmul(user_matrix, u_features)  # [num_users, dim]
        u_pre = u_pre.squeeze()
        return u_pre


class DualGNN(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(DualGNN, self).__init__(config, dataloader)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = config.get('k', 40)
        self.aggr_mode = config.get('aggr_mode', 'mean')
        self.user_aggr_mode = config.get('user_aggr_mode', 'softmax')
        self.num_layer = 1
        self.construction = config.get('construction', 'weighted_sum')
        self.reg_weight = config['reg_weight']
        self.drop_rate = config.get('drop_rate', 0.1)
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)

        # 使用缓存管理器load或build user-user 图
        data_path = config['data_path']
        dataset_name = config['dataset']
        cache_manager = GraphCacheManager(data_path, dataset_name)
        preprocessor = DualGNNPreprocessor(cache_manager)
        
        # 尝试从缓存load（需要转换格式）
        user_graph_dict_raw = preprocessor.load_user_graph(self.k, self.construction)
        
        if user_graph_dict_raw is None:
            # build新的 user-user 图
            print('Building user-user graph...')
            train_interactions = dataloader.dataset.inter_matrix(form='coo').astype(np.float32)
            user_graph_dict_raw = preprocessor.build_user_user_graph(
                train_interactions, 
                k=self.k, 
                construction=self.construction
            )
            # save到缓存
            preprocessor.save_user_graph(user_graph_dict_raw, self.k, self.construction)
            print(f'Built user-user graph with {len(user_graph_dict_raw)} users')
        
        # Convert to MMRec-tmp format
        self.user_graph_dict = {}
        for u, neighbors in user_graph_dict_raw.items():
            # For simplicity, all weights are 1.0
            self.user_graph_dict[u] = [neighbors, [1.0] * len(neighbors)]

        # packing interaction in training into edge_index
        train_interactions = dataloader.dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i.data, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        for i in range(self.num_item):
            self.item_index[i] = i

        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv, dtype=torch.int64).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt, dtype=torch.int64).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        if self.visual_feat is not None:
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx), self.visual_feat.size(1)).to(self.device)
            self.v_gcn = GCN(dataloader.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device, features=self.visual_feat)
        if self.textual_feat is not None:
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx), self.textual_feat.size(1)).to(self.device)
            self.t_gcn = GCN(dataloader.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                         device=self.device, features=self.textual_feat)

        self.user_graph = User_Graph_sample(num_user, self.user_aggr_mode, self.dim_latent)

        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def pre_epoch_processing(self):
        # Match MMRec-tmp: resample every epoch
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        representation = None
        if self.visual_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.visual_feat)
            representation = self.v_rep
        if self.textual_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.textual_feat)
            if representation is None:
                representation = self.t_rep
            else:
                representation += self.t_rep

        # multi-modal information aggregation
        if self.construction == 'weighted_sum':
            if self.v_rep is not None and self.t_rep is not None:
                user_rep = torch.matmul(torch.cat((self.v_rep[:self.num_user].unsqueeze(2), 
                                                   self.t_rep[:self.num_user].unsqueeze(2)), dim=2),
                                        self.weight_u)
                user_rep = torch.squeeze(user_rep)
            elif self.v_rep is not None:
                user_rep = self.v_rep[:self.num_user]
            else:
                user_rep = self.t_rep[:self.num_user]
        else:
            user_rep = representation[:self.num_user]
        
        item_rep = representation[self.num_user:]
        
        # User graph propagation
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
        user_rep = user_rep + h_u1
        
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        
        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        
        loss_value = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        
        if self.construction == 'weighted_sum':
            reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
            reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        
        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k
            user_graph_index.append(user_graph_sample)

        return user_graph_index, user_weight_matrix

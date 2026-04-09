# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
LOBSTER: Bilateral global semantic enhancement for multimedia recommendation, Information Fusion, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import random

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class LOBSTER(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LOBSTER, self).__init__(config, dataset)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.num_modal = 3
        self.emb_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.factor_num_u = config['factor_num_u']
        self.factor_num_i = config['factor_num_i']
        self.n_nodes = self.n_users + self.n_items

        self.user_embds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.emb_size)))
        self.item_embds = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.emb_size)))

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.masked_adj = None
        self.pruning_random = False

        self.factor_u = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.factor_num_u, self.emb_size)))
        self.factor_i = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.factor_num_i, self.emb_size)))

        self.models = nn.ModuleDict({
            'id_model': LayerGCN(self.n_users, self.n_items, self.user_embds, self.item_embds, self.emb_size, self.factor_u, self.factor_i),
            'v_model': LayerGCN(self.n_users, self.n_items, self.user_embds, self.visual_feat, self.emb_size, self.factor_u, self.factor_i),
            't_model': LayerGCN(self.n_users, self.n_items, self.user_embds, self.textual_feat, self.emb_size, self.factor_u, self.factor_i),
        })

        self.reg_loss = L2Loss()
        self.mf_loss = BPRLoss()


    def pre_epoch_processing(self):
        if self.dropout > 0.0:
            keep_len = int(self.edge_values.size(0) * (1. - self.dropout))
            if self.pruning_random:
                keep_idx = torch.tensor(random.sample(range(self.edge_values.size(0)), keep_len))
            else:
                keep_idx = torch.multinomial(self.edge_values, keep_len)
            self.pruning_random = True ^ self.pruning_random
            keep_indices = self.edge_indices[:, keep_idx]
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            all_values = torch.cat((keep_values, keep_values))
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
            self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj_matrix.shape).to(self.device)
        else:
            self.masked_adj = self.norm_adj_matrix

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_ego_embeddings(self):
        ego_embeddings = torch.cat([self.user_embds, self.item_embds], 0)
        return ego_embeddings

    def forward(self, adj):
        user_id, item_id = self.models['id_model'](adj)
        user_v, item_v = self.models['v_model'](adj)
        user_t, item_t = self.models['t_model'](adj)
        u_embeddings = torch.cat([user_id, user_v, user_t], -1)
        i_embeddings = torch.cat([item_id, item_v, item_t], -1)
        return u_embeddings, i_embeddings

    def bpr_loss(self, u_embeddings, i_embeddings, user, pos_item, neg_item):
        u_embeddings = u_embeddings[user]
        posi_embeddings = i_embeddings[pos_item]
        negi_embeddings = i_embeddings[neg_item]
        modality_weights = self.Cannikin_Law_weight(u_embeddings, posi_embeddings, negi_embeddings)
        pos_scores = torch.mul(torch.mul(u_embeddings, posi_embeddings), modality_weights).sum(dim=1)
        neg_scores = torch.mul(torch.mul(u_embeddings, negi_embeddings), modality_weights).sum(dim=1)
        bpr_loss = -torch.sum(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def emb_loss(self, user, pos_item, neg_item):
        u_ego_embeddings = self.user_embds[user]
        posi_ego_embeddings = self.item_embds[pos_item]
        negi_ego_embeddings = self.item_embds[neg_item]
        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        return reg_loss

    def Cannikin_Law_weight(self, user_e, pos_e, neg_e):
        pos_score_ = torch.mul(user_e, pos_e).view(-1, self.num_modal, self.emb_size).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, self.num_modal, self.emb_size).sum(dim=-1)
        modality_indicator = 1 - (pos_score_ - neg_score_).softmax(-1).detach()
        modality_weights = torch.tile(modality_indicator.view(-1, self.num_modal, 1), [1, 1, self.emb_size])
        modality_weights = modality_weights.view(-1, self.num_modal * self.emb_size)
        return modality_weights

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        user_all_embeddings, item_all_embeddings = self.forward(self.masked_adj)
        mf_loss = self.bpr_loss(user_all_embeddings, item_all_embeddings, user, pos_item, neg_item)
        reg_loss = self.emb_loss(user, pos_item, neg_item)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj_matrix)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

class LayerGCN(nn.Module):
    def __init__(self, num_user, num_item, user_fea, item_fea, emb_size, global_embedding_u, global_embedding_i):
        super(LayerGCN, self).__init__()
        self.n_layers = 4
        self.num_user = num_user
        self.num_item = num_item
        self.user_fea = user_fea
        self.item_fea = item_fea
        self.emb_size = emb_size
        self.global_embedding_u = global_embedding_u
        self.global_embedding_i = global_embedding_i
        self.mlp = nn.Sequential(nn.Linear(self.item_fea.shape[1], self.emb_size), nn.Tanh())

    def forward(self, adj):
        global_embd_u = torch.sum(self.global_embedding_u, 0)
        global_embd_i = torch.sum(self.global_embedding_i, 0)
        user_embd = self.user_fea + global_embd_u[None, :]
        item_embd = self.mlp(self.item_fea) + global_embd_i[None, :]
        ego_embeddings = torch.cat((user_embd, item_embd), dim=0)
        all_embeddings = ego_embeddings
        embeddings_layers = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings)
            embeddings_layers.append(all_embeddings)
        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0)
        u_embd, i_embd = torch.split(ui_all_embeddings, [self.num_user, self.num_item])
        return u_embd, i_embd

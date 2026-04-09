# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
Mining Latent Structures for Multimedia Recommendation, ACM MM, 2021
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_cache import GraphCacheManager


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class LATTICE(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(LATTICE, self).__init__(config, dataloader)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config.get('feat_embed_dim', config['embedding_size'])
        self.weight_size = config.get('weight_size', [64, 64])
        self.knn_k = config.get('knn_k', 10)
        self.lambda_coeff = config.get('lambda_coeff', 0.9)
        self.cf_model = config.get('cf_model', 'lightgcn')
        self.n_layers = config.get('n_layers', 1)
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True

        # load dataset info
        self.interaction_matrix = dataloader.dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        self.item_adj = None

        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if config.get('cf_model', 'lightgcn') == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            dropout_list = config.get('mess_dropout', [0.1, 0.1])
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        # Use cache manager
        cache_manager = GraphCacheManager(config['data_path'], config['dataset'])
        cache_dir = cache_manager.get_model_cache_dir('LATTICE')
        image_adj_file = os.path.join(cache_dir, 'image_adj_{}.pt'.format(self.knn_k))
        text_adj_file = os.path.join(cache_dir, 'text_adj_{}.pt'.format(self.knn_k))

        if self.visual_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
            if os.path.exists(image_adj_file):
                print(f'Loading image_adj from {image_adj_file}')
                image_adj = torch.load(image_adj_file)
            else:
                print('Building image_adj...')
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
                image_adj = compute_normalized_laplacian(image_adj)
                os.makedirs(os.path.dirname(image_adj_file), exist_ok=True)
                torch.save(image_adj, image_adj_file)
                print(f'Saved image_adj to {image_adj_file}')
            self.image_original_adj = image_adj.cuda()

        if self.textual_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)
            if os.path.exists(text_adj_file):
                print(f'Loading text_adj from {text_adj_file}')
                text_adj = torch.load(text_adj_file)
            else:
                print('Building text_adj...')
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
                text_adj = compute_normalized_laplacian(text_adj)
                os.makedirs(os.path.dirname(text_adj_file), exist_ok=True)
                torch.save(text_adj, text_adj_file)
                print(f'Saved text_adj to {text_adj_file}')
            self.text_original_adj = text_adj.cuda()

        if self.visual_feat is not None:
            self.image_trs = nn.Linear(self.visual_feat.shape[1], self.feat_embed_dim)
        if self.textual_feat is not None:
            self.text_trs = nn.Linear(self.textual_feat.shape[1], self.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def pre_epoch_processing(self):
        self.build_item_graph = True

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, build_item_graph=False):
        if self.visual_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.textual_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
        
        if build_item_graph:
            weight = self.softmax(self.modal_weight)

            if self.visual_feat is not None:
                self.image_adj = build_sim(image_feats)
                self.image_adj = build_knn_neighbourhood(self.image_adj, topk=self.knn_k)
                learned_adj = self.image_adj
                original_adj = self.image_original_adj
            if self.textual_feat is not None:
                self.text_adj = build_sim(text_feats)
                self.text_adj = build_knn_neighbourhood(self.text_adj, topk=self.knn_k)
                learned_adj = self.text_adj
                original_adj = self.text_original_adj
            if self.visual_feat is not None and self.textual_feat is not None:
                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
                original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj

            learned_adj = compute_normalized_laplacian(learned_adj)
            if self.item_adj is not None:
                del self.item_adj
            self.item_adj = (1 - self.lambda_coeff) * learned_adj + self.lambda_coeff * original_adj
        
        # 使用 detach 避免重复反向传播
        item_adj_detach = self.item_adj.detach() if not build_item_graph else self.item_adj

        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.mm(item_adj_detach, h)

        if self.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif self.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.norm_adj, build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        return batch_mf_loss + self.reg_weight * (torch.norm(u_g_embeddings) + torch.norm(pos_i_g_embeddings) + torch.norm(neg_i_g_embeddings))

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj, build_item_graph=False)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

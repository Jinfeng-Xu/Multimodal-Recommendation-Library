# coding: utf-8
# @email: enoche.chow@gmail.com
r"""

################################################
paper:  Bootstrap Latent Representations for Multi-modal Recommendation
https://arxiv.org/abs/2207.05969
"""
import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class BM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(BM3, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        nn.init.xavier_normal_(self.predictor.weight)

        if self.visual_feat is not None:
            self.visual_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
            self.visual_trs = nn.Linear(self.visual_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.visual_trs.weight)
        if self.textual_feat is not None:
            self.textual_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)
            self.textual_trs = nn.Linear(self.textual_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.textual_trs.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # online network
        u_online_ori, i_online_ori = self.forward()
        textual_feat_online, visual_feat_online = None, None
        if self.textual_feat is not None:
            textual_feat_online = self.textual_trs(self.textual_embedding.weight)
        if self.visual_feat is not None:
            visual_feat_online = self.visual_trs(self.visual_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.textual_feat is not None:
                textual_feat_target = textual_feat_online.clone()
                textual_feat_target = F.dropout(textual_feat_target, self.dropout)

            if self.visual_feat is not None:
                visual_feat_target = visual_feat_online.clone()
                visual_feat_target = F.dropout(visual_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        loss_textual, loss_visual, loss_textual_visual, loss_visual_textual = 0.0, 0.0, 0.0, 0.0
        if self.textual_feat is not None:
            textual_feat_online = self.predictor(textual_feat_online)
            textual_feat_online = textual_feat_online[items, :]
            textual_feat_target = textual_feat_target[items, :]
            loss_textual = 1 - cosine_similarity(textual_feat_online, i_target.detach(), dim=-1).mean()
            loss_textual_visual = 1 - cosine_similarity(textual_feat_online, textual_feat_target.detach(), dim=-1).mean()
        if self.visual_feat is not None:
            visual_feat_online = self.predictor(visual_feat_online)
            visual_feat_online = visual_feat_online[items, :]
            visual_feat_target = visual_feat_target[items, :]
            loss_visual = 1 - cosine_similarity(visual_feat_online, i_target.detach(), dim=-1).mean()
            loss_visual_textual = 1 - cosine_similarity(visual_feat_online, visual_feat_target.detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_textual + loss_visual + loss_textual_visual + loss_visual_textual).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui


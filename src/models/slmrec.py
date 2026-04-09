# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
Self-Supervised Learning for Multimedia Recommendation, IEEE TMM, 2022
"""

import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
from torch_scatter import scatter
from sklearn.cluster import KMeans

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class SLMRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(SLMRec, self).__init__(config, dataloader)
        
        self.a_feat = None  # no audio feature
        self.config = config
        self.infonce_criterion = nn.CrossEntropyLoss()
        self.__init_weight(dataloader)

    def __init_weight(self, dataloader):
        self.num_users = self.n_users
        self.num_items = self.n_items
        self.latent_dim = self.config.get('recdim', self.config.get('embedding_size', 64))
        self.n_layers = self.config.get('layer_num', self.config.get('n_layers', 3))
        self.mm_fusion_mode = self.config.get('mm_fusion_mode', 'concat')
        self.temp = self.config.get('temp', 0.2)

        self.create_u_embeding_i()

        self.all_items = self.all_users = None

        train_interactions = dataloader.dataset.inter_matrix(form='csr').astype(np.float32)
        coo = self.create_adj_mat(train_interactions).tocoo()
        indices = torch.LongTensor([coo.row.tolist(), coo.col.tolist()])
        self.norm_adj = torch.sparse.FloatTensor(indices, torch.FloatTensor(coo.data), coo.shape)
        self.norm_adj = self.norm_adj.to(self.device)
        self.f = nn.Sigmoid()

        if self.config.get("ssl_task", "FM") == "FAC":
            # Fine and Coarse
            self.g_i_iv = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_v_iv = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_iv_iva = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_a_iva = nn.Linear(self.latent_dim, self.latent_dim)
            self.g_iva_ivat = nn.Linear(self.latent_dim, self.latent_dim // 2)
            self.g_t_ivat = nn.Linear(self.latent_dim, self.latent_dim // 2)
            nn.init.xavier_uniform_(self.g_i_iv.weight)
            nn.init.xavier_uniform_(self.g_v_iv.weight)
            nn.init.xavier_uniform_(self.g_iv_iva.weight)
            nn.init.xavier_uniform_(self.g_a_iva.weight)
            nn.init.xavier_uniform_(self.g_iva_ivat.weight)
            nn.init.xavier_uniform_(self.g_t_ivat.weight)
            self.ssl_temp = self.config.get("ssl_temp", 0.2)
        elif self.config.get("ssl_task", "FM") in ["FD", "FD+FM"]:
            # Feature dropout
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.ssl_temp = self.config.get("ssl_temp", 0.2)
            self.dropout_rate = self.config.get("dropout_rate", 0.3)
            self.dropout = nn.Dropout(p=self.dropout_rate)
        elif self.config.get("ssl_task", "FM") == "FM":
            # Feature Masking
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.ssl_temp = self.config.get("ssl_temp", 0.2)

    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.visual_feat is not None:
            self.v_dense_emb = self.v_dense(self.visual_feat)
        if self.textual_feat is not None:
            self.t_dense_emb = self.t_dense(self.textual_feat)

        def compute_graph(u_emb, i_emb):
            all_emb = torch.cat([u_emb, i_emb])
            embs = [all_emb]
            g_droped = self.norm_adj
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            return light_out

        self.i_emb = compute_graph(users_emb, items_emb)
        self.i_emb_u, self.i_emb_i = torch.split(self.i_emb, [self.num_users, self.num_items])
        
        if self.visual_feat is not None:
            self.v_emb = compute_graph(users_emb, self.v_dense_emb)
            self.v_emb_u, self.v_emb_i = torch.split(self.v_emb, [self.num_users, self.num_items])
        
        if self.textual_feat is not None:
            self.t_emb = compute_graph(users_emb, self.t_dense_emb)
            self.t_emb_u, self.t_emb_i = torch.split(self.t_emb, [self.num_users, self.num_items])

        # multi - modal features fusion
        if self.visual_feat is not None and self.textual_feat is not None:
            user = self.embedding_user_after_GCN(self.mm_fusion([self.i_emb_u, self.v_emb_u, self.t_emb_u]))
            item = self.embedding_item_after_GCN(self.mm_fusion([self.i_emb_i, self.v_emb_i, self.t_emb_i]))
        elif self.visual_feat is not None:
            user = self.embedding_user_after_GCN(self.mm_fusion([self.i_emb_u, self.v_emb_u]))
            item = self.embedding_item_after_GCN(self.mm_fusion([self.i_emb_i, self.v_emb_i]))
        else:
            user = self.i_emb_u
            item = self.i_emb_i

        return user, item

    def feature_masking(self, users_idx, items_idx, dropout=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        rand_range = 3 if self.textual_feat is not None else 2
        rand_idx1 = np.random.randint(rand_range)
        rand_idx2 = 0
        while True:
            rand_idx2 = np.random.randint(rand_range)
            if rand_idx2 != rand_idx1:
                break

        v_dense = self.v_dense_emb if self.visual_feat is not None else None
        t_dense = self.t_dense_emb if self.textual_feat is not None else None

        def compute_graph(u_emb, i_emb, idx):
            all_emb_1 = torch.cat([u_emb,
                                   i_emb if rand_idx1 != idx else torch.zeros((self.num_items, self.latent_dim)).to(self.device)])
            all_emb_2 = torch.cat([u_emb,
                                   i_emb if rand_idx2 != idx else torch.zeros((self.num_items, self.latent_dim)).to(self.device)])
            ego_emb_sub_1 = all_emb_1
            ego_emb_sub_2 = all_emb_2
            embs_sub_1 = [ego_emb_sub_1]
            embs_sub_2 = [ego_emb_sub_2]
            g_droped = self.norm_adj

            for _ in range(self.n_layers):
                ego_emb_sub_1 = torch.sparse.mm(g_droped, ego_emb_sub_1)
                ego_emb_sub_2 = torch.sparse.mm(g_droped, ego_emb_sub_2)
                if dropout:
                    ego_emb_sub_1 = self.dropout(ego_emb_sub_1)
                    ego_emb_sub_2 = self.dropout(ego_emb_sub_2)
                embs_sub_2.append(ego_emb_sub_1)
                embs_sub_1.append(ego_emb_sub_2)
            embs_sub_1 = torch.stack(embs_sub_1, dim=1)
            embs_sub_2 = torch.stack(embs_sub_2, dim=1)

            light_out_sub_1 = torch.mean(embs_sub_1, dim=1)
            light_out_sub_2 = torch.mean(embs_sub_2, dim=1)

            users_sub_1, items_sub_1 = torch.split(light_out_sub_1, [self.num_users, self.num_items])
            users_sub_2, items_sub_2 = torch.split(light_out_sub_2, [self.num_users, self.num_items])
            return users_sub_1[users_idx], items_sub_1[items_idx], users_sub_2[users_idx], items_sub_2[items_idx]

        i_emb_u_sub_1, i_emb_i_sub_1, i_emb_u_sub_2, i_emb_i_sub_2 = compute_graph(users_emb, items_emb, idx=2 if t_dense is not None else 1)
        v_emb_u_sub_1, v_emb_i_sub_1, v_emb_u_sub_2, v_emb_i_sub_2 = compute_graph(users_emb, v_dense, idx=0) if v_dense is not None else (None, None, None, None)
        t_emb_u_sub_1, t_emb_i_sub_1, t_emb_u_sub_2, t_emb_i_sub_2 = compute_graph(users_emb, t_dense, idx=2) if t_dense is not None else (None, None, None, None)

        reps_1 = [i_emb_u_sub_1]
        reps_2 = [i_emb_u_sub_2]
        if v_emb_u_sub_1 is not None:
            reps_1.append(v_emb_u_sub_1)
            reps_2.append(v_emb_u_sub_2)
        if t_emb_u_sub_1 is not None:
            reps_1.append(t_emb_u_sub_1)
            reps_2.append(t_emb_u_sub_2)

        users_sub_1 = self.embedding_user_after_GCN(self.mm_fusion(reps_1))
        users_sub_2 = self.embedding_user_after_GCN(self.mm_fusion(reps_2))

        reps_1_item = [i_emb_i_sub_1]
        reps_2_item = [i_emb_i_sub_2]
        if v_emb_i_sub_1 is not None:
            reps_1_item.append(v_emb_i_sub_1)
            reps_2_item.append(v_emb_i_sub_2)
        if t_emb_i_sub_1 is not None:
            reps_1_item.append(t_emb_i_sub_1)
            reps_2_item.append(t_emb_i_sub_2)

        items_sub_1 = self.embedding_item_after_GCN(self.mm_fusion(reps_1_item))
        items_sub_2 = self.embedding_item_after_GCN(self.mm_fusion(reps_2_item))

        users_sub_1 = torch.nn.functional.normalize(users_sub_1, dim=1)
        users_sub_2 = torch.nn.functional.normalize(users_sub_2, dim=1)
        items_sub_1 = torch.nn.functional.normalize(items_sub_1, dim=1)
        items_sub_2 = torch.nn.functional.normalize(items_sub_2, dim=1)

        logits_user = torch.mm(users_sub_1, users_sub_2.T)
        logits_user /= self.ssl_temp
        labels_user = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
        ssl_loss_user = self.ssl_criterion(logits_user, labels_user)

        logits_item = torch.mm(items_sub_1, items_sub_2.T)
        logits_item /= self.ssl_temp
        labels_item = torch.tensor(list(range(items_sub_2.shape[0]))).to(self.device)
        ssl_loss_item = self.ssl_criterion(logits_item, labels_item)

        return ssl_loss_user + ssl_loss_item

    def fac(self, idx):
        x_i_iv = self.g_i_iv(self.i_emb_i[idx])
        x_v_iv = self.g_v_iv(self.v_emb_i[idx])
        v_logits = torch.mm(x_i_iv, x_v_iv.T)

        v_logits /= self.ssl_temp
        v_labels = torch.tensor(list(range(x_i_iv.shape[0]))).to(self.device)
        v_loss = self.infonce_criterion(v_logits, v_labels)
        
        if self.textual_feat is not None:
            x_iv_iva = self.g_iv_iva(x_i_iv)
            x_iva_ivat = self.g_iva_ivat(x_iv_iva)
            x_t_ivat = self.g_t_ivat(self.t_emb_i[idx])

            t_logits = torch.mm(x_iva_ivat, x_t_ivat.T)
            t_logits /= self.ssl_temp
            t_labels = torch.tensor(list(range(x_iva_ivat.shape[0]))).to(self.device)
            t_loss = self.infonce_criterion(t_logits, t_labels)

            return v_loss + t_loss
        else:
            return v_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        self.all_users, self.all_items = self.compute()
        users_emb = self.all_users[users]
        items_emb = self.all_items
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def getEmbedding(self, users, pos_items, neg_items):
        self.all_users, self.all_items = self.compute()
        users_emb = self.all_users[users]
        pos_emb = self.all_items[pos_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)

        if neg_items is None:
            neg_emb_ego = neg_emb = None
        else:
            neg_emb = self.all_items[neg_items]
            neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def calculate_loss(self, interaction):
        users, pos = interaction[0], interaction[1]
        main_loss = self.infonce(users, pos)
        ssl_loss = self.compute_ssl(users, pos)
        ssl_alpha = self.config.get('ssl_alpha', 0.1)
        return main_loss + ssl_alpha * ssl_loss

    def compute_ssl(self, users, items):
        ssl_task = self.config.get("ssl_task", "FM")
        if ssl_task == "FAC":
            return self.fac(items)
        elif ssl_task == "FD":
            return self.feature_masking(users.long(), items.long())
        elif ssl_task == "FM":
            return self.feature_masking(users.long(), items.long())
        elif ssl_task == "FD+FM":
            return self.feature_masking(users.long(), items.long(), dropout=True)
        return 0.0

    def infonce(self, users, pos):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), None)
        users_emb = torch.nn.functional.normalize(users_emb, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, dim=1)
        logits = torch.mm(users_emb, pos_emb.T)
        logits /= self.temp
        labels = torch.tensor(list(range(users_emb.shape[0]))).to(self.device)
        return self.infonce_criterion(logits, labels)

    def create_u_embeding_i(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        init_type = self.config.get("init", "xavier")
        if init_type == "xavier":
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        mul_modal_cnt = 0
        if self.visual_feat is not None:
            self.visual_feat = torch.nn.functional.normalize(self.visual_feat, dim=1)
            self.v_dense = nn.Linear(self.visual_feat.shape[1], self.latent_dim)
            nn.init.xavier_uniform_(self.v_dense.weight)
            mul_modal_cnt += 1
        if self.textual_feat is not None:
            self.textual_feat = torch.nn.functional.normalize(self.textual_feat, dim=1)
            self.t_dense = nn.Linear(self.textual_feat.shape[1], self.latent_dim)
            nn.init.xavier_uniform_(self.t_dense.weight)
            mul_modal_cnt += 1

        self.item_feat_dim = self.latent_dim * (mul_modal_cnt + 1)

        self.embedding_item_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        self.embedding_user_after_GCN = nn.Linear(self.item_feat_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_item_after_GCN.weight)
        nn.init.xavier_uniform_(self.embedding_user_after_GCN.weight)

    def create_adj_mat(self, interaction_csr):
        user_np, item_np = interaction_csr.nonzero()
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        adj_type = self.config.get('adj_type', 'pre')
        if adj_type == 'plain':
            adj_matrix = adj_mat
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
        elif adj_type == 'pre':
            rowsum = np.array(adj_mat.sum(1)) + 1e-08
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])

        return adj_matrix

    def mm_fusion(self, reps: list):
        if self.mm_fusion_mode == "concat":
            z = torch.cat(reps, dim=1)
        elif self.mm_fusion_mode == "mean":
            z = torch.mean(torch.stack(reps), dim=0)
        return z

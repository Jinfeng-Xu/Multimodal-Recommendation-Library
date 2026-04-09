# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
Hypercomplex Prompt-aware Multimodal Recommendation, CIKM, 2025
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from utils.graph_cache import GraphCacheManager


class CayleyDicksonMult(nn.Module):
    def __init__(self, algebra_dim):
        super(CayleyDicksonMult, self).__init__()
        self.algebra_dim = algebra_dim
        if algebra_dim > 2:
            self.sub_mult = CayleyDicksonMult(algebra_dim // 2)

    def forward(self, h1, h2):
        if self.algebra_dim == 2:
            real1, imag1 = torch.chunk(h1, 2, dim=1)
            real2, imag2 = torch.chunk(h2, 2, dim=1)

            real = real1 * real2 - imag1 * imag2
            imag = real1 * imag2 + imag1 * real2
            return torch.cat((real, imag), dim=1)

        else:
            a, b = torch.chunk(h1, 2, dim=1)
            c, d = torch.chunk(h2, 2, dim=1)

            sub_real = self.sub_mult(a, c) - self.sub_mult(d, b)
            sub_imag = self.sub_mult(a, d) + self.sub_mult(b, c)

            return torch.cat((sub_real, sub_imag), dim=1)


class HypercomplexGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', device=None):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out


class HypercomplexGNN(torch.nn.Module):
    def __init__(self, num_user, num_item, embed_dim, num_layers=None, device=None, features=None, algebra_dim=None):
        super(HypercomplexGNN, self).__init__()

        self.num_layers = num_layers
        self.dim_latent = embed_dim
        self.dim_feat = features.size(1)
        self.aggr_mode = 'add'
        self.num_user = num_user
        self.num_item = num_item
        self.num_nodes = num_user + num_item
        self.device = device
        self.algebra_dim = algebra_dim

        self.emb_s = nn.Parameter(torch.FloatTensor(self.num_user, self.dim_latent)).to(self.device)
        self.emb_x = [nn.Parameter(torch.FloatTensor(self.num_user, self.dim_latent)).to(self.device) for _ in
                      range(self.algebra_dim - 1)]

        nn.init.xavier_uniform_(self.emb_s)
        for i in range(self.algebra_dim - 1):
            nn.init.xavier_uniform_(self.emb_x[i])

        self.prompt_emb = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user + num_item, self.dim_latent * self.algebra_dim), dtype=torch.float32,
            requires_grad=True),
            gain=1).to(self.device))
        self.gnn_layers = None

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.gnn_layers = nn.ModuleList([
                HypercomplexGNNLayer(embed_dim, embed_dim, aggr='add')
                for _ in range(algebra_dim)
            ])

        else:
            self.gnn_layers = nn.ModuleList([
                HypercomplexGNNLayer(embed_dim, embed_dim, aggr='add')
                for _ in range(algebra_dim)
            ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_index, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        features_s = F.normalize(torch.cat((self.emb_s, temp_features), dim=0)).to(self.device)
        features = features_s
        for i in range(self.algebra_dim - 1):
            features_x = F.normalize(torch.cat((self.emb_x[i], temp_features), dim=0)).to(self.device)
            features = torch.cat((features, features_x), dim=1)
        h_components = torch.chunk(features, self.algebra_dim, dim=1)

        linear_total = []
        for n in range(self.num_layers):
            linear_outputs = []
            for i in range(self.algebra_dim):
                features_out = self.gnn_layers[i](h_components[i], edge_index)
                h_components[i].data = features_out
                linear_outputs.append(features_out)
            linear_combined = torch.cat(linear_outputs, dim=1)
            linear_total.append(linear_combined)

        linear_total = torch.sum(torch.stack(linear_total, dim=0), dim=0)
        combined = linear_total + self.prompt_emb

        total = []
        total.append(self.emb_s)
        for i in range(self.algebra_dim - 1):
            total.append(self.emb_x[i])

        return combined, total


class HPMRec(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(HPMRec, self).__init__(config, dataloader)

        self.ssl_weight = config.get('ssl_weight', 0.1)
        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']
        dim_x = config['embedding_size']
        self.feat_embed_dim = config.get('feat_embed_dim', 64)
        self.n_mm_layers = config.get('n_mm_layers', 2)
        self.num_layers = config.get('n_layers', 2)
        self.knn_k = config.get('knn_k', 10)
        self.mm_image_weight = config.get('mm_image_weight', 0.5)

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config.get('aggr_mode', 'add')
        self.user_aggr_mode = 'softmax'
        self.dataset = dataloader.dataset
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.v_rep = None
        self.t_rep = None
        self.id_rep = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.algebra_dim = config.get('algebra_dim', 2)
        self.cayley_mult = CayleyDicksonMult(algebra_dim=self.algebra_dim)
        self.mm_adj = None

        # Use cache manager for mm_adj
        cache_manager = GraphCacheManager(config['data_path'], config['dataset'])
        cache_dir = cache_manager.get_model_cache_dir('HPMRec')
        mm_adj_file = os.path.join(cache_dir, 'mm_adj_{}.pt'.format(self.knn_k))

        if self.visual_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
            self.image_trs = nn.Linear(self.visual_feat.shape[1], self.feat_embed_dim)
        if self.textual_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)
            self.text_trs = nn.Linear(self.textual_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            print(f'Loading mm_adj from {mm_adj_file}')
            self.mm_adj = torch.load(mm_adj_file)
        else:
            print('Building mm_adj...')
            if self.visual_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.textual_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.visual_feat is not None and self.textual_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(mm_adj_file), exist_ok=True)
            torch.save(self.mm_adj, mm_adj_file)
            print(f'Saved mm_adj to {mm_adj_file}')

        train_interactions = dataloader.dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        if self.visual_feat is not None:
            self.v_gcn = HypercomplexGNN(num_user=num_user, num_item=num_item, embed_dim=self.dim_latent,
                                         num_layers=self.num_layers,
                                         device=self.device, features=self.visual_feat, algebra_dim=self.algebra_dim)

        if self.textual_feat is not None:
            self.t_gcn = HypercomplexGNN(num_user=num_user, num_item=num_item, embed_dim=self.dim_latent,
                                         num_layers=self.num_layers,
                                         device=self.device, features=self.textual_feat, algebra_dim=self.algebra_dim)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = HypercomplexGNN(num_user=num_user, num_item=num_item, embed_dim=self.dim_latent,
                                      num_layers=self.num_layers,
                                      device=self.device, features=self.id_feat, algebra_dim=self.algebra_dim)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        if self.id_feat is not None:
            self.id_rep, self.id_preference = self.id_gcn(self.edge_index, self.id_feat)

        if self.visual_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.visual_feat)

        if self.textual_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.textual_feat)

        representation_idv = self.cayley_mult(self.id_rep, self.v_rep)
        representation_idt = self.cayley_mult(self.id_rep, self.t_rep)

        self.v_rep = torch.add(self.v_rep, 0.1 * representation_idv)
        self.t_rep = torch.add(self.t_rep, 0.1 * representation_idt)

        representation = torch.cat((self.id_rep, self.v_rep, self.t_rep), dim=1)

        user_rep = representation[:self.n_users]
        item_rep = representation[self.n_users:]

        h = item_rep
        for i in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)

        main_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        reg_loss = self._calculate_reg_loss(interaction)
        ssl_loss = self._calculate_ssl_loss()

        total_loss = main_loss + reg_loss + self.ssl_weight * ssl_loss
        return total_loss

    def _calculate_reg_loss(self, interaction):
        user = interaction[0]
        reg_embedding_loss_v = 0.0
        reg_embedding_loss_t = 0.0
        reg_embedding_loss_id = 0.0
        for i in range(self.algebra_dim):
            reg_embedding_loss_v += (self.v_preference[i][user] ** 2).mean() if self.v_preference is not None else 0.0
            reg_embedding_loss_t += (self.t_preference[i][user] ** 2).mean() if self.t_preference is not None else 0.0
            reg_embedding_loss_id += (
                    self.id_preference[i][user] ** 2).mean() if self.id_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t + reg_embedding_loss_id)
        return reg_loss

    def _calculate_ssl_loss(self):
        ssl_loss = 0.0

        if self.visual_feat is not None and self.textual_feat is not None:
            id_rep = self.id_rep[self.n_users:]
            v_rep = self.v_rep[self.n_users:]
            t_rep = self.t_rep[self.n_users:]
            ssl_loss += torch.mean(torch.abs(id_rep - t_rep))
            ssl_loss += torch.mean(torch.abs(id_rep - t_rep))
            ssl_loss += torch.mean(torch.abs(v_rep - t_rep))

        id_real, id_imag = self.decompose_hypercomplex(self.id_rep)
        v_real, v_imag = self.decompose_hypercomplex(self.v_rep)
        t_real, t_imag = self.decompose_hypercomplex(self.t_rep)
        ssl_loss += -torch.mean(torch.abs(id_real - id_imag))
        ssl_loss += -torch.mean(torch.abs(v_real - v_imag))
        ssl_loss += -torch.mean(torch.abs(t_real - t_imag))

        return ssl_loss

    def decompose_hypercomplex(self, hyper_embed):
        chunks = torch.chunk(hyper_embed, self.algebra_dim, dim=1)
        real = chunks[0]
        imaginary = torch.mean(torch.stack(chunks[1:], dim=1), dim=1)
        return real, imaginary


    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

        return score_matrix

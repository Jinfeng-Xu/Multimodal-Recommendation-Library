import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import torch_sparse

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from utils.utils import (
    build_sim, compute_normalized_laplacian, build_knn_neighbourhood,
    build_knn_normalized_graph
)
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
from utils.graph_cache import GraphCacheManager


class COHESION(GeneralRecommender):
    def __init__(self, config, dataset):
        super(COHESION, self).__init__(config, dataset)
        self.setup_parameters(config, dataset)

    def setup_parameters(self, config, dataset):
        # Initialize parameters and embeddings
        self.n_nodes = self.n_users + self.n_items
        self.dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.k = 40
        self.num_layer = config['num_layer']
        self.dataset = dataset
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep, self.t_rep, self.id_rep = None, None, None
        self.v_preference, self.t_preference, self.id_preference = None, None, None
        self.dim_latent = 64
        self.mm_adj = None

        # Use cache manager for user-graph and mm_adj
        cache_manager = GraphCacheManager(config['data_path'], config['dataset'])
        cache_dir = cache_manager.get_model_cache_dir('COHESION')
        
        user_graph_file = os.path.join(cache_dir, 'user_graph_dict.pt')
        mm_adj_file = os.path.join(cache_dir, f'mm_adj_{self.knn_k}.pt')
        
        # Load user graph dictionary
        if os.path.exists(user_graph_file):
            self.user_graph_dict = torch.load(user_graph_file)
        else:
            # Build user-user graph (same as MENTOR)
            from collections import defaultdict
            import torch as torch_lib
            train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
            edge_dict = defaultdict(set)
            user_edges = zip(*train_interactions.nonzero())
            for user, item in user_edges:
                edge_dict[user].add(item)
            
            user_graph_matrix = np.zeros((self.n_users, self.n_users))
            key_list = sorted(list(edge_dict.keys()))
            for head in range(len(key_list)):
                for rear in range(head+1, len(key_list)):
                    head_key = key_list[head]
                    rear_key = key_list[rear]
                    inter_len = len(edge_dict[head_key].intersection(edge_dict[rear_key]))
                    if inter_len > 0:
                        user_graph_matrix[head_key][rear_key] = inter_len
                        user_graph_matrix[rear_key][head_key] = inter_len
            
            self.user_graph_dict = {}
            for i in range(self.n_users):
                user_num_i = len(torch_lib.nonzero(torch_lib.from_numpy(user_graph_matrix[i])))
                if user_num_i <= 200:
                    user_i = torch_lib.topk(torch_lib.from_numpy(user_graph_matrix[i]), int(user_num_i))
                else:
                    user_i = torch_lib.topk(torch_lib.from_numpy(user_graph_matrix[i]), 200)
                edge_list_i = user_i.indices.numpy().tolist()
                edge_list_j = user_i.values.numpy().tolist()
                self.user_graph_dict[i] = [edge_list_i, edge_list_j]
            
            torch_lib.save(self.user_graph_dict, user_graph_file)

        # Load or generate mm_adj
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            self.image_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
            self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj
            cache_manager.save_graph('COHESION', 'mm_adj', self.mm_adj)

        # Create user and item embeddings
        self.initialize_embeddings(config, dataset)

        # Construct interaction edge
        self.train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)

        # Create normalized adjacency matrices
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = self.norm_adj

        # Create GCN layers for different modalities
        self.create_gcn_layers()

        # Create user graph and result embeddings
        self.user_graph = User_Graph_sample(self.n_users, self.dim_latent)
        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_nodes, self.dim)))
        ).to(self.device)

    def load_or_generate_mm_adj(self, mm_adj_file):
        if os.path.exists(mm_adj_file):
            return torch.load(mm_adj_file)
        else:
            mm_adj = self.generate_mm_adj()
            torch.save(mm_adj, mm_adj_file)
            return mm_adj

    def generate_mm_adj(self):
        image_adj, text_adj = None, None
        if self.visual_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
        if self.textual_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())

        if self.visual_feat is not None and self.textual_feat is not None:
            mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj, image_adj
        else:
            mm_adj = image_adj if image_adj is not None else text_adj

        return mm_adj

    def initialize_embeddings(self, config, dataset):
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        if self.visual_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
        if self.textual_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)

        self.weight_u = nn.Parameter(
            nn.init.xavier_normal_(
                torch.tensor(np.random.randn(self.n_users, 2, 1), dtype=torch.float32, requires_grad=True))
        )
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

    def create_gcn_layers(self):
        if self.visual_feat is not None:
            self.v_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                                  device=self.device, features=self.visual_feat)
        if self.textual_feat is not None:
            self.t_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                                  device=self.device, features=self.textual_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCNLayer(self.n_users, self.n_items, num_layer=self.num_layer, dim_latent=64,
                               device=self.device, features=self.id_feat)

    def get_edge_info(self):
        rows = torch.from_numpy(self.train_interactions.row)
        cols = torch.from_numpy(self.train_interactions.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

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

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.train_interactions
        inter_M_t = self.train_interactions.transpose()
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

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # get representation and id_rep_data
        representation, id_rep_data = self.build_representation()

        # get user and item representation
        user_rep, item_rep = self.process_user_item_representation(representation, id_rep_data)

        # get user and item tensor
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]

        # Adaptively optimize the weight of the three modalities
        adaptive_weight = self.adaptive_optimization(user_tensor, pos_item_tensor, neg_item_tensor)
        pos_scores = torch.sum(user_tensor * pos_item_tensor * adaptive_weight, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor * adaptive_weight, dim=1)
        return pos_scores, neg_scores

    def build_representation(self):
        id_rep, id_preference = self.id_gcn(self.id_feat, self.id_feat, self.masked_adj)
        id_rep_data = id_rep.data

        representation = id_rep_data

        if self.visual_feat is not None:
            self.v_rep, self.v_preference = self.v_gcn(self.visual_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.v_rep), dim=1)

        if self.textual_feat is not None:
            self.t_rep, self.t_preference = self.t_gcn(self.textual_feat, self.id_feat, self.masked_adj)
            representation = torch.cat((id_rep_data, self.t_rep) if representation is None
                                       else (id_rep_data, self.v_rep, self.t_rep), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        id_rep_data = torch.unsqueeze(id_rep_data, 2)

        return representation, id_rep_data

    def process_user_item_representation(self, representation, id_rep_data):
        user_rep, item_rep = None, None

        if self.v_rep is not None and self.t_rep is not None:
            user_rep = torch.cat((id_rep_data[:self.n_users], self.v_rep[:self.n_users], self.t_rep[:self.n_users]),
                                 dim=2)
            user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

        item_rep = representation[self.n_users:]

        h_i = item_rep
        for i in range(self.n_layers):
            h_i = torch.sparse.mm(self.mm_adj, h_i)
        h_u = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)

        user_rep = user_rep + h_u
        item_rep = item_rep + h_i

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return user_rep, item_rep

    def adaptive_optimization(self, user_e, pos_e, neg_e):
        pos_score_ = torch.mul(user_e, pos_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        neg_score_ = torch.mul(user_e, neg_e).view(-1, 3, self.dim_latent).sum(dim=-1)
        modality_indicator = 1 - (pos_score_ - neg_score_).softmax(-1).detach()

        adaptive_weight = torch.tile(modality_indicator.view(-1, 3, 1), [1, 1, self.dim_latent])
        adaptive_weight = adaptive_weight.view(-1, 3 * self.dim_latent)

        return adaptive_weight

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
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
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class User_Graph_sample(torch.nn.Module):
    """
        user-user graph
    """

    def __init__(self, num_user, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCNLayer(torch.nn.Module):
    def __init__(self, num_user, num_item, num_layer, dim_latent=None, device=None, features=None):
        super(GCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.preference = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = torch.abs(
            ((torch.mul(id_embd, id_embd) + torch.mul(temp_features, temp_features)) / 2) + 1e-8).sqrt()
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        ego_embeddings = x
        all_embeddings = ego_embeddings.to(self.device)
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings).to(self.device)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1).to(self.device)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings).to(self.device)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0).to(self.device)

        return ui_all_embeddings, self.preference


class BGCNLayer(torch.nn.Module):
    """
        basic layer-refined GCN
    """

    def __init__(self, num_user, num_item, num_layer, dim_latent=None, device=None, features=None):
        super(BGCNLayer, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.num_layer = num_layer
        self.device = device
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features, id_embd, adj):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        ego_embeddings = x
        all_embeddings = ego_embeddings.to(self.device)
        embeddings_layers = [all_embeddings]

        for layer_idx in range(self.num_layer):
            all_embeddings = torch.sparse.mm(adj, all_embeddings).to(self.device)
            _weights = F.cosine_similarity(all_embeddings, ego_embeddings, dim=-1).to(self.device)
            all_embeddings = torch.einsum('a,ab->ab', _weights, all_embeddings).to(self.device)
            embeddings_layers.append(all_embeddings)

        ui_all_embeddings = torch.sum(torch.stack(embeddings_layers, dim=0), dim=0).to(self.device)

        return ui_all_embeddings, self.preference

# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
Multi-Modal Self-Supervised Learning for Recommendation, WWW, 2023
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
from torch import autograd
from scipy.sparse import csr_matrix

from common.abstract_recommender import GeneralRecommender


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.G_drop1 = 0.31
        self.G_drop2 = 0.5
        # Eq(4)
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim / 4)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 4)),
            nn.Dropout(self.G_drop1),

            nn.Linear(int(dim / 4), int(dim / 8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim / 8)),
            nn.Dropout(self.G_drop2),

            nn.Linear(int(dim / 8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = 100 * self.net(x.float())
        return output.view(-1)


class MMSSL(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(MMSSL, self).__init__(config, dataloader)
        
        self.dim_E = config['embedding_size']
        self.mmlayer = config.get('mmlayer', 3)
        self.weight_size = [64] * self.mmlayer
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.dim_E] + self.weight_size
        self.reg_weight = config['reg_weight']
        self.tau = config.get('ssl_temp', 0.5)
        self.feat_reg_decay = 1e-5
        self.gene_u, self.gene_real, self.gene_fake = None, None, {}
        self.log_log_scale = 0.00001
        self.real_data_tau = 0.005
        self.ui_pre_scale = 100
        self.gp_rate = 1
        self.T = 1
        self.m_topk_rate = 0.0001
        self.cl_rate = config.get('ssl_alpha', 0.003)
        self.G_rate = config.get('G_rate', 0.0001)
        self.model_cat_rate = 0.55
        self.id_cat_rate = 0.36
        self.sparse = 1
        
        # build交互矩阵
        edge_index = self._build_edge_index(dataloader)
        adjusted_item_ids = edge_index[:, 1] - self.n_users
        interaction_matrix = sp.coo_matrix((np.ones(len(edge_index)),
                                            (edge_index[:, 0], adjusted_item_ids)),
                                           shape=(self.n_users, self.n_items), dtype=np.float32)
        interaction_matrix = interaction_matrix.tocsr()
        self.ui_graph = self.ui_graph_raw = interaction_matrix
        self.iu_graph = self.ui_graph.T
        
        dense_matrix = interaction_matrix.todense()
        self.image_ui_graph_tmp = self.text_ui_graph_tmp = torch.tensor(dense_matrix).to(self.device)
        self.image_iu_graph_tmp = self.text_iu_graph_tmp = torch.tensor(dense_matrix.T).to(self.device)
        
        self.image_ui_index = {'x': [], 'y': []}
        self.text_ui_index = {'x': [], 'y': []}
        
        self.ui_graph = self.matrix_to_tensor(self.csr_norm(self.ui_graph, mean_flag=True))
        self.iu_graph = self.matrix_to_tensor(self.csr_norm(self.iu_graph, mean_flag=True))
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        # Discriminator
        self.D = Discriminator(self.n_items).to(self.device)
        self.D.apply(self.weights_init)

        # Modality encoders
        self.image_trans = nn.Linear(self.visual_feat.shape[1], self.dim_E)
        self.text_trans = nn.Linear(self.textual_feat.shape[1], self.dim_E)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)

        self.encoder = nn.ModuleDict()
        self.encoder['image_encoder'] = self.image_trans
        self.encoder['text_encoder'] = self.text_trans

        self.common_trans = nn.Linear(self.dim_E, self.dim_E)
        nn.init.xavier_uniform_(self.common_trans.weight)
        self.align = nn.ModuleDict()
        self.align['common_trans'] = self.common_trans

        # User and item embeddings
        self.user_id_embedding = nn.Embedding(self.n_users, self.dim_E)
        self.item_id_embedding = nn.Embedding(self.n_items, self.dim_E)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_feats = self.visual_feat
        self.text_feats = self.textual_feat

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm1d(self.dim_E)

        # Multi-head self-attention
        self.head_num = 4
        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_k': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_v': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([self.dim_E, self.dim_E]))),
            'w_self_attention_cat': nn.Parameter(
                initializer(torch.empty([self.head_num * self.dim_E, self.dim_E]))),
        })

        self.embedding_dict = {'user': {}, 'item': {}}

    def _build_edge_index(self, dataloader):
        """build边索引"""
        interactions = []
        for batch in dataloader:
            users = batch[0].cpu().numpy()
            items = batch[1].cpu().numpy()
            for u, i in zip(users, items):
                interactions.append([u, i + self.n_users])
        return torch.tensor(interactions, dtype=torch.long).cpu().numpy()

    def mm(self, x, y):
        if self.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum + 1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum + 1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag * csr_mat * colsum_diag
        else:
            return rowsum_diag * csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))
        values = torch.from_numpy(cur_matrix.data)
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse_coo_tensor(indices, values, shape).to(torch.float32).to(self.device)

    def para_dict_to_tenser(self, para_dict):
        tensors = []
        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)
        return tensors

    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):
        # trans_w: weight dictionary for Q, K, V，包括用于查询（Q）、键（K）和值（V）的转换权重，以及其他自注意力相关权重
        # 将input的两组嵌入转换为张量
        q = self.para_dict_to_tenser(embedding_t)  # query  (2, num, dim_E)
        v = k = self.para_dict_to_tenser(embedding_t_1)  # key and value  (2, num, dim_E)

        # compute每个头的维度
        beh, N, d_h = q.shape[0], q.shape[1], self.dim_E / self.head_num

        # compute查询、键和值
        Q = torch.matmul(q, trans_w['w_q'])  # query与其权重的矩阵乘法
        K = torch.matmul(k, trans_w['w_k'])  # Matrix multiplication of key with weights
        V = v  # value  (2, num, dim_E)

        # Reshape Q, K for multi-head  [4,2,N,16]
        Q = Q.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)
        K = K.reshape(beh, N, self.head_num, int(d_h)).permute(2, 0, 1, 3)

        # Add dimension for broadcasting
        Q = torch.unsqueeze(Q, 2)  # (self.head_num, 2, num, 1, d_h)
        K = torch.unsqueeze(K, 1)  # (self.head_num, 2, 1, num, d_h)
        V = torch.unsqueeze(V, 1)  # (2, 1, num, dim_E)

        # compute注意力权重
        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  # Dot-product attention  (self.head_num, 2, num, num)
        att = torch.sum(att, dim=-1)
        att = torch.unsqueeze(att, dim=-1)
        att = F.softmax(att, dim=2)  # Get attention distribution with softmax

        # Apply attention weights to values
        Z = torch.mul(att, V)  # (self.head_num, 2, num, d_h)
        Z = torch.sum(Z, dim=2)

        # Concatenate results from different heads
        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])  # (2, num, dim_E)

        # Apply normalization and scaling
        Z = self.model_cat_rate * F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):
        image_feats = self.encoder['image_encoder'](self.image_feats)
        text_feats = self.encoder['text_encoder'](self.text_feats)

        image_feats = self.align['common_trans'](image_feats)
        text_feats = self.align['common_trans'](text_feats)

        for i in range(self.mmlayer):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)
            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)

        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id

        user_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'],
                                                   self.embedding_dict['user'])
        item_z, _ = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'],
                                                   self.embedding_dict['item'])

        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)

        u_g_embeddings = self.user_id_embedding.weight + self.id_cat_rate * F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + self.id_cat_rate * F.normalize(item_emb, p=2, dim=1)

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]

        for i in range(self.n_ui_layers):
            if i == (self.n_ui_layers - 1):
                u_g_embeddings = self.softmax(torch.mm(ui_graph, i_g_embeddings))
                i_g_embeddings = self.softmax(torch.mm(iu_graph, u_g_embeddings))
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings)
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings)

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0)

        u_g_embeddings = u_g_embeddings + self.model_cat_rate * F.normalize(image_user_feats, p=2, dim=1) + \
                         self.model_cat_rate * F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + self.model_cat_rate * F.normalize(image_item_feats, p=2, dim=1) + \
                         self.model_cat_rate * F.normalize(text_item_feats, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, \
            text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id

    def batched_contrastive_loss(self, z1, z2, batch_size=1024):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)

        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            tmp_i = indices[i * batch_size:(i + 1) * batch_size]

            tmp_refl_sim_list = []
            tmp_between_sim_list = []
            for j in range(num_batches):
                tmp_j = indices[j * batch_size:(j + 1) * batch_size]
                tmp_refl_sim = f(self.sim(z1[tmp_i], z1[tmp_j]))
                tmp_between_sim = f(self.sim(z1[tmp_i], z2[tmp_j]))

                tmp_refl_sim_list.append(tmp_refl_sim)
                tmp_between_sim_list.append(tmp_between_sim)

            refl_sim = torch.cat(tmp_refl_sim_list, dim=-1)
            between_sim = torch.cat(tmp_between_sim_list, dim=-1)

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() / (
                    refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:,
                                                           i * batch_size:(i + 1) * batch_size].diag()) + 1e-8))

            del refl_sim, between_sim, tmp_refl_sim_list, tmp_between_sim_list

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds, \
            _, _, _, _, _, _ = self.forward(self.ui_graph, self.iu_graph, self.image_ui_graph,
                                            self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # BPR loss
        pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1)
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=1)
        regularizer = 1. / 2 * (u_g_embeddings ** 2).sum() + 1. / 2 * (pos_i_g_embeddings ** 2).sum() + 1. / 2 * (
                    neg_i_g_embeddings ** 2).sum()
        regularizer = regularizer / len(users)

        mf_loss = -torch.mean(torch.log(self.sigmoid(pos_scores - neg_scores) + 1e-10))
        emb_loss = self.reg_weight * regularizer

        # Contrastive learning loss
        cl_loss = self.batched_contrastive_loss(image_item_embeds, text_item_embeds)

        # Feature regularization
        feat_reg = 1. / 2 * (image_item_embeds ** 2).sum() + 1. / 2 * (text_item_embeds ** 2).sum()
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = self.feat_reg_decay * feat_reg

        return mf_loss + emb_loss + self.cl_rate * cl_loss + feat_emb_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]

        ua_embeddings, ia_embeddings, _, _, _, _, _, _, _, _, _, _ = self.forward(
            self.ui_graph, self.iu_graph, self.image_ui_graph,
            self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)

        u_embeddings = ua_embeddings[users]
        scores = torch.matmul(u_embeddings, ia_embeddings.transpose(0, 1))
        return scores

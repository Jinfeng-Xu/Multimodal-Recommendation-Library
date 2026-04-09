# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback, ACM MM, 2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr="mean", **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels

        self.out_channels = out_channels

    def forward(self, x, edge_index, weight_vector, size=None):
        self.weight_vector = weight_vector

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j * self.weight_vector

    def update(self, aggr_out):
        return aggr_out


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, self_loops=False):
        super(GATConv, self).__init__(aggr="add")

        self.self_loops = self_loops

        self.in_channels = in_channels

        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, size_i, edge_index_i):
        self.alpha = torch.mul(x_i, x_j).sum(dim=-1)

        self.alpha = F.leaky_relu(self.alpha)

        self.alpha = softmax(self.alpha, edge_index_i, num_nodes=size_i)

        return x_j * self.alpha.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class EGCN(torch.nn.Module):
    def __init__(self, num_user, num_item, dim_E, aggr_mode, has_act, has_norm):
        super(EGCN, self).__init__()

        self.num_user = num_user

        self.num_item = num_item

        self.dim_E = dim_E

        self.aggr_mode = aggr_mode

        self.has_act = has_act

        self.has_norm = has_norm

        self.id_embedding = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_E))))

        self.conv_embed_1 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

        self.conv_embed_2 = SAGEConv(dim_E, dim_E, aggr=aggr_mode)

    def forward(self, edge_index, weight_vector):
        x = self.id_embedding

        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

        if self.has_norm:
            x = F.normalize(x)

        x_hat_1 = self.conv_embed_1(x, edge_index, weight_vector)

        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)

        x_hat_2 = self.conv_embed_2(x_hat_1, edge_index, weight_vector)

        if self.has_act:
            x_hat_2 = F.leaky_relu_(x_hat_2)

        return x + x_hat_1 + x_hat_2


class CGCN(torch.nn.Module):
    def __init__(
        self,
        features,
        num_user,
        num_item,
        dim_C,
        aggr_mode,
        num_routing,
        has_act,
        has_norm,
        is_word=False,
        device="cpu",
    ):
        super(CGCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.num_routing = num_routing
        self.has_act = has_act
        self.has_norm = has_norm
        self.dim_C = dim_C
        self.device = device
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_C)))).to(device)
        self.conv_embed_1 = GATConv(self.dim_C, self.dim_C)
        self.is_word = is_word

        if is_word:
            self.word_tensor = torch.LongTensor(features).to(device)
            self.features = nn.Embedding(torch.max(features[1]) + 1, dim_C).to(device)
            nn.init.xavier_normal_(self.features.weight)
        else:
            self.dim_feat = features.size(1)
            self.features = features
            self.MLP = nn.Linear(self.dim_feat, self.dim_C)
            nn.init.xavier_normal_(self.MLP.weight)

    def forward(self, edge_index):
        if self.is_word:
            features = self.features(self.word_tensor[1])
        else:
            features = F.leaky_relu(self.MLP(self.features))

        if self.has_norm:
            preference = F.normalize(self.preference)
            features = F.normalize(features)

        for i in range(self.num_routing):
            x = torch.cat((preference, features), dim=0)
            x_hat_1 = self.conv_embed_1(x, edge_index)
            preference = preference + x_hat_1[: self.num_user]
            if self.has_norm:
                preference = F.normalize(preference)

        x = torch.cat((preference, features), dim=0)
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
        x_hat_1 = self.conv_embed_1(x, edge_index)
        if self.has_act:
            x_hat_1 = F.leaky_relu_(x_hat_1)
        return x + x_hat_1, self.conv_embed_1.alpha.view(-1, 1)


class GRCN(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(GRCN, self).__init__(config, dataloader)

        self.num_user = self.n_users

        self.num_item = self.n_items

        dim_x = config["embedding_size"]

        dim_C = config.get("latent_embedding", 16)

        num_layer = config.get("n_layers", 2)

        self.aggr_mode = "add"

        self.weight_mode = "confid"

        self.fusion_mode = "concat"

        has_act = False

        has_norm = True

        is_word = False

        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)

        self.reg_weight = config["reg_weight"]

        self.dropout = 0.0

        self.pruning = True

        # build杈圭储寮?
        train_interactions = dataloader.dataset.inter_matrix(form="coo").astype(np.float32)

        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)

        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.num_modal = 0
        num_model = 0
        self.id_gcn = EGCN(self.num_user, self.num_item, dim_x, self.aggr_mode, has_act, has_norm)

        if self.visual_feat is not None:
            self.v_gcn = CGCN(
                self.visual_feat,
                self.num_user,
                self.num_item,
                dim_C,
                self.aggr_mode,
                num_layer,
                has_act,
                has_norm,
                is_word,
                self.device,
            )
            num_model += 1

        if self.textual_feat is not None:
            self.t_gcn = CGCN(
                self.textual_feat,
                self.num_user,
                self.num_item,
                dim_C,
                self.aggr_mode,
                num_layer,
                has_act,
                has_norm,
                is_word,
                self.device,
            )
            num_model += 1

        self.model_specific_conf = nn.Parameter(
            nn.init.xavier_normal_(torch.rand((self.num_user + self.num_item, num_model)))
        )
        self.result = nn.init.xavier_normal_(torch.rand((self.num_user + self.num_item, dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self):
        weight = None
        content_rep = None
        num_modal = 0
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)

        if self.visual_feat is not None:
            num_modal += 1
            v_rep, weight_v = self.v_gcn(edge_index)
            weight = weight_v
            content_rep = v_rep

        if self.textual_feat is not None:
            num_modal += 1
            t_rep, weight_t = self.t_gcn(edge_index)
            if weight is None:
                weight = weight_t
                content_rep = t_rep
            else:
                content_rep = torch.cat((content_rep, t_rep), dim=1)
                if self.weight_mode == "mean":
                    weight = weight + weight_t
                else:
                    weight = torch.cat((weight, weight_t), dim=1)

        if self.weight_mode == "mean":
            weight = weight / num_modal
        elif self.weight_mode == "max":
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)
        elif self.weight_mode == "confid":
            confidence = torch.cat(
                (self.model_specific_conf[edge_index[0]], self.model_specific_conf[edge_index[1]]), dim=0
            )
            weight = weight * confidence
            weight, _ = torch.max(weight, dim=1)
            weight = weight.view(-1, 1)

        if self.pruning:
            weight = torch.relu(weight)

        id_rep = self.id_gcn(edge_index, weight)

        if self.fusion_mode == "concat":
            representation = torch.cat((id_rep, content_rep), dim=1)
        elif self.fusion_mode == "id":
            representation = id_rep
        elif self.fusion_mode == "mean":
            representation = (id_rep + content_rep) / 2

        self.result = representation
        return representation

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users
        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(score[:, 0] - score[:, 1]) + 1e-8) + 1e-8)
        reg_loss = self.reg_weight * (
            out[batch_users].pow(2).mean() + out[pos_items].pow(2).mean() + out[neg_items].pow(2).mean()
        )
        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        batch_users = interaction[0]
        out = self.forward()
        user_out = out[batch_users]
        item_out = out[self.n_users :]
        return torch.matmul(user_out, item_out.t())

        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)) + 1e-10))

        reg_embedding_loss = (
            self.id_gcn.id_embedding[user_tensor] ** 2 + self.id_gcn.id_embedding[item_tensor] ** 2
        ).mean()
        if self.visual_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference**2).mean()
        if self.textual_feat is not None:
            reg_embedding_loss += (self.t_gcn.preference**2).mean()

        reg_content_loss = torch.zeros(1).to(self.device)
        if self.visual_feat is not None:
            reg_content_loss = reg_content_loss + (self.v_gcn.preference[user_tensor] ** 2).mean()
        if self.textual_feat is not None:
            reg_content_loss = reg_content_loss + (self.t_gcn.preference[user_tensor] ** 2).mean()

        reg_confid_loss = (self.model_specific_conf**2).mean()
        reg_loss = reg_embedding_loss + reg_content_loss
        reg_loss = self.reg_weight * reg_loss
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result[: self.n_users]
        item_tensor = self.result[self.n_users :]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
r"""
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback, AAAI, 2016
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class VBPR(GeneralRecommender):
    r"""VBPR is a basic multimodal recommendation model that uses visual features to enhance item representation
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)

        # loadparameter
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']

        # Define embeddings and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        
        # Automatically use available modality features (vision or text)
        if self.visual_feat is not None and self.textual_feat is not None:
            self.item_raw_features = torch.cat((self.textual_feat, self.visual_feat), -1)
        elif self.visual_feat is not None:
            self.item_raw_features = self.visual_feat
        else:
            self.item_raw_features = self.textual_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameterinitialize
        self.apply(xavier_uniform_initialization)

    def get_user_embedding(self, user):
        r""" Get user embedding tensor
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get item embedding tensor
        """
        return self.item_embedding[item, :]

    def forward(self, dropout=0.0):
        item_embeddings = self.item_linear(self.item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        score_batch = torch.mm(user_embeddings[user, :], item_embeddings.t())
        return score_batch

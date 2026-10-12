# coding: utf-8
"""Information-Guided Selective Modality-Interest Alignment (CIKM 2026).

Adapted from https://github.com/Wenze1/AMUR for MRLib's unified trainer.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_ops import chunked_knn_graph, normalized_ui_graph
from utils.graph_cache import GraphCacheManager


class _GraphLearner(nn.Module):
    def __init__(self, dimension, candidate_graph, dropout):
        super().__init__()
        candidate_graph = candidate_graph.coalesce()
        self.register_buffer("indices", candidate_graph.indices())
        self.register_buffer("prior", candidate_graph.values())
        self.shape = candidate_graph.shape
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dimension + 1, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )

    def forward(self, item_embeddings, residual):
        source, target = self.indices
        edge_input = torch.cat(
            [item_embeddings[source], item_embeddings[target], self.prior[:, None]], dim=1
        )
        probabilities = torch.sigmoid(self.edge_mlp(edge_input).squeeze(1))
        hard = torch.where(probabilities > 0.5, probabilities, torch.zeros_like(probabilities))
        selected = hard.detach() - probabilities.detach() + probabilities
        values = residual * self.prior + (1.0 - residual) * selected
        graph = torch.sparse_coo_tensor(
            self.indices, values, self.shape, device=item_embeddings.device
        ).coalesce()
        return graph, probabilities


class AMUR(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if not hasattr(self, "visual_feat") or not hasattr(self, "textual_feat"):
            raise ValueError("AMUR requires visual and textual item features")

        self.dimension = int(config["embedding_size"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.n_layers = int(config["n_layers"])
        self.reg_weight = float(config["reg_weight"])
        self.cl_weight = float(config["cl_ip_loss"])
        self.ip_weight = float(config["cl_ip_loss"])
        self.ui_cl_weight = float(config["uicl_loss"])
        self.kl_weight = float(config["kl_loss"])
        self.residual = float(config["res_conn"])
        self.temperature = float(config["temperature"])

        interaction = dataset.inter_matrix(form="coo").astype(np.float32)
        norm_adj, interaction_graph = normalized_ui_graph(interaction, self.device)
        self.register_buffer("norm_adj", norm_adj)
        self.register_buffer("interaction_graph", interaction_graph)

        self.user_embedding = nn.Embedding(self.n_users, self.dimension)
        self.item_embedding = nn.Embedding(self.n_items, self.dimension)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        cache = GraphCacheManager(config["data_path"], config["dataset"]).get_model_cache_dir("AMUR")
        image_graph = self._load_graph(
            cache, "visual", self.visual_feat, int(config["knn_k"]), int(config["knn_chunk_size"])
        )
        text_graph = self._load_graph(
            cache, "text", self.textual_feat, int(config["knn_k"]), int(config["knn_chunk_size"])
        )
        self.image_learner = _GraphLearner(self.dimension, image_graph, float(config["drop"]))
        self.text_learner = _GraphLearner(self.dimension, text_graph, float(config["drop"]))

        self.image_gate = nn.Sequential(
            nn.Linear(3 * self.dimension, self.dimension), nn.GELU(),
            nn.Dropout(float(config["drop"])), nn.Linear(self.dimension, self.dimension), nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(3 * self.dimension, self.dimension), nn.GELU(),
            nn.Dropout(float(config["drop"])), nn.Linear(self.dimension, self.dimension), nn.Sigmoid()
        )

    @staticmethod
    def _load_graph(cache, name, features, topk, chunk_size):
        path = os.path.join(cache, f"{name}_knn_{topk}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location=features.device).coalesce()
        graph = chunked_knn_graph(features, topk, chunk_size)
        torch.save(graph.cpu(), path)
        return graph

    def _ui_embeddings(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [embeddings]
        for _ in range(self.n_ui_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            layers.append(embeddings)
        return torch.stack(layers, dim=1).mean(dim=1)

    def _modality_embeddings(self, graph, item_embeddings):
        for _ in range(self.n_layers):
            item_embeddings = torch.sparse.mm(graph, item_embeddings)
        users = torch.sparse.mm(self.interaction_graph, item_embeddings)
        return torch.cat([users, item_embeddings], dim=0), item_embeddings

    def forward(self, train=False):
        content = self._ui_embeddings()
        content_items = content[self.n_users :]
        image_graph, image_prob = self.image_learner(self.item_embedding.weight, self.residual)
        text_graph, text_prob = self.text_learner(self.item_embedding.weight, self.residual)
        image_all, image_items = self._modality_embeddings(image_graph, self.item_embedding.weight)
        text_all, text_items = self._modality_embeddings(text_graph, self.item_embedding.weight)
        side = (0.1 * image_all + 0.9 * text_all) / 3.0
        final = content + side
        users, items = torch.split(final, [self.n_users, self.n_items], dim=0)
        if not train:
            return users, items
        image_shared = self.image_gate(torch.cat([image_items, text_items, content_items], dim=1)) * image_items
        text_shared = self.text_gate(torch.cat([text_items, image_items, content_items], dim=1)) * text_items
        return users, items, side, image_shared, text_shared, image_prob, text_prob

    def _info_nce(self, first, second):
        first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
        logits = first @ second.T / self.temperature
        labels = torch.arange(first.shape[0], device=first.device)
        return F.cross_entropy(logits, labels)

    @staticmethod
    def _edge_kl(probability, prior):
        probability = probability.clamp(1e-7, 1.0 - 1e-7)
        target = (prior / prior.max().clamp_min(1e-8)).clamp(1e-7, 1.0 - 1e-7)
        return F.binary_cross_entropy(probability, target)

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        user_all, item_all, side, image_shared, text_shared, image_prob, text_prob = self.forward(True)
        user_batch, positive_batch, negative_batch = (
            user_all[users], item_all[positive], item_all[negative]
        )
        recommendation = -F.logsigmoid(
            (user_batch * positive_batch).sum(1) - (user_batch * negative_batch).sum(1)
        ).mean()
        regularization = self.reg_weight * (
            user_batch.square().sum() + positive_batch.square().sum() + negative_batch.square().sum()
        ) / (2.0 * max(1, users.shape[0]))
        alignment = 0.5 * (
            self._info_nce(image_shared[positive], text_shared[positive])
            + self._info_nce(text_shared[positive], image_shared[positive])
        )
        interest = F.mse_loss(image_shared[positive], text_shared[positive])
        side_items = side[self.n_users :]
        ui_alignment = self._info_nce(side_items[positive], user_batch)
        information = self._edge_kl(image_prob, self.image_learner.prior) + self._edge_kl(
            text_prob, self.text_learner.prior
        )
        return (
            recommendation + regularization + self.cl_weight * alignment
            + self.ip_weight * interest + self.ui_cl_weight * ui_alignment
            + self.kl_weight * information
        )

    def full_sort_predict(self, interaction):
        users, items = self.forward(False)
        return users[interaction[0]] @ items.T

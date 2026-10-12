# coding: utf-8
"""Popular-Niche Wavelet Graph Learning for Multimodal Recommendation.

Adapted from https://github.com/orangeheyue/PNGRec for MRLib.
"""

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_cache import GraphCacheManager
from utils.popular_niche_graph import PopularNicheGraph


def _scipy_to_torch(matrix, device):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce().to(device)


def _chunked_knn_graph(features, topk, chunk_size=1024):
    """Construct a normalized sparse KNN graph without an O(N^2) allocation."""
    features = F.normalize(features, dim=1)
    rows, columns, weights = [], [], []
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        similarity = features[start:stop] @ features.T
        values, indices = torch.topk(similarity, k=min(topk, features.shape[0]), dim=1)
        rows.append(
            torch.arange(start, stop, device=features.device)[:, None].expand_as(indices).reshape(-1)
        )
        columns.append(indices.reshape(-1))
        weights.append(values.reshape(-1).clamp_min(1e-7))
    row = torch.cat(rows)
    column = torch.cat(columns)
    value = torch.cat(weights)
    degree = torch.zeros(features.shape[0], device=features.device)
    degree.scatter_add_(0, row, value)
    value = value * degree[row].clamp_min(1e-12).rsqrt() * degree[column].clamp_min(1e-12).rsqrt()
    return torch.sparse_coo_tensor(
        torch.stack([row, column]), value, (features.shape[0], features.shape[0])
    ).coalesce()


class _ItemwiseCrossModalAttention(nn.Module):
    """Linear-memory cross-modal attention for large item catalogs."""

    def __init__(self, dimension):
        super().__init__()
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)

    def forward(self, visual, textual):
        score = (self.query(textual) * self.key(visual)).sum(dim=1, keepdim=True)
        score = torch.sigmoid(score / math.sqrt(visual.shape[1]))
        return 0.5 * (textual + score * self.value(visual))


class PNGRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.visual_feat is None or self.textual_feat is None:
            raise ValueError("PNGRec requires both visual_feat and textual_feat")

        self.embedding_dim = int(config["embedding_size"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.n_layers = int(config["n_layers"])
        self.reg_weight = float(config["reg_weight"])
        self.cl_weight = float(config["cl_loss"])
        self.temperature = float(config["temperature"])
        self.use_png = bool(config["USE_PNG"])
        self.use_pnq = bool(config["USE_PNQ"])
        self.use_pnl = bool(config["USE_PNL"])

        interaction = dataset.inter_matrix(form="coo").astype(np.float32)
        adjacency = sp.bmat(
            [[None, interaction], [interaction.T, None]], format="coo", dtype=np.float32
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        nonzero = degree > 0
        inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
        normalized = (sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocsr()
        self.register_buffer("norm_adj", _scipy_to_torch(normalized, self.device))
        normalized_interaction = normalized[: self.n_users, self.n_users :]
        self.register_buffer("interaction_graph", _scipy_to_torch(normalized_interaction, self.device))

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.visual_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=False)
        self.visual_projection = nn.Linear(self.visual_feat.shape[1], self.embedding_dim)
        self.text_projection = nn.Linear(self.textual_feat.shape[1], self.embedding_dim)
        self.cross_modal_attention = _ItemwiseCrossModalAttention(self.embedding_dim)

        self.visual_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.fusion_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.interest_attention = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 2),
        )
        self.modal_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(float(config["dropout_rate"]))

        cache_manager = GraphCacheManager(config["data_path"], config["dataset"])
        cache_dir = cache_manager.get_model_cache_dir("PNGRec")
        self.visual_graph = self._load_or_build_modality_graph(
            cache_dir, "visual", self.visual_feat, int(config["image_knn_k"]), config
        )
        self.text_graph = self._load_or_build_modality_graph(
            cache_dir, "text", self.textual_feat, int(config["text_knn_k"]), config
        )
        self.fusion_graph = self._fuse_graphs(self.visual_graph, self.text_graph)

        wavelet = str(config["wavelet"])
        level = int(config["wavelet_level"])
        png_cache = os.path.join(cache_dir, f"popular_niche_{wavelet}_{level}.pt")
        if os.path.exists(png_cache):
            payload = torch.load(png_cache, map_location=self.device)
            popular, niche, quadrants = payload["popular"], payload["niche"], payload["quadrants"]
        else:
            builder = PopularNicheGraph(
                normalized_interaction,
                wavelet,
                level,
                self.device,
                batch_size=int(config["wavelet_batch_size"]),
            )
            popular, niche, quadrants = builder.build()
            torch.save(
                {
                    "popular": popular.cpu(),
                    "niche": niche.cpu(),
                    "quadrants": {key: value.cpu() for key, value in quadrants.items()},
                },
                png_cache,
            )
        self.register_buffer("popular_graph", popular.to(self.device).coalesce())
        self.register_buffer("niche_graph", niche.to(self.device).coalesce())
        for name, graph in quadrants.items():
            self.register_buffer(f"quadrant_{name}", graph.to(self.device).coalesce())

    def _load_or_build_modality_graph(self, cache_dir, name, features, topk, config):
        path = os.path.join(cache_dir, f"{name}_knn_{topk}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location=self.device).coalesce()
        graph = _chunked_knn_graph(
            features, topk, chunk_size=int(config["knn_chunk_size"])
        )
        torch.save(graph.cpu(), path)
        return graph

    @staticmethod
    def _row_scale(graph, scale):
        graph = graph.coalesce()
        index = graph.indices()
        return torch.sparse_coo_tensor(
            index, graph.values() * scale[index[0]], graph.shape, device=graph.device
        ).coalesce()

    def _fuse_graphs(self, visual, text):
        visual_degree = torch.sparse.sum(visual, dim=1).to_dense()
        text_degree = torch.sparse.sum(text, dim=1).to_dense()
        visual_weight = visual_degree / (visual_degree + text_degree + 1e-8)
        return (self._row_scale(visual, visual_weight) + self._row_scale(text, 1.0 - visual_weight)).coalesce()

    def _ui_propagation(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [embeddings]
        for _ in range(self.n_ui_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            layers.append(embeddings)
        return torch.stack(layers, dim=1).mean(dim=1)

    def _item_propagation(self, graph, item_embeddings):
        for _ in range(self.n_layers):
            item_embeddings = torch.sparse.mm(graph, item_embeddings)
        user_embeddings = torch.sparse.mm(self.interaction_graph, item_embeddings)
        return torch.cat([user_embeddings, item_embeddings], dim=0)

    def _interest_embeddings(self, graph, item_embeddings):
        user_embeddings = torch.sparse.mm(graph, item_embeddings)
        return torch.cat([user_embeddings, item_embeddings], dim=0)

    def forward(self, train=False):
        visual = self.visual_projection(self.visual_embedding.weight)
        textual = self.text_projection(self.text_embedding.weight)
        fused = self.cross_modal_attention(visual, textual)
        visual_items = self.item_embedding.weight * self.visual_gate(visual)
        text_items = self.item_embedding.weight * self.text_gate(textual)
        fusion_items = self.item_embedding.weight * self.fusion_gate(fused)

        content = self._ui_propagation()
        visual_embeddings = self._item_propagation(self.visual_graph, visual_items)
        text_embeddings = self._item_propagation(self.text_graph, text_items)
        fusion_embeddings = self._item_propagation(self.fusion_graph, fusion_items)
        popular = self._interest_embeddings(self.popular_graph, fusion_items)
        niche = self._interest_embeddings(self.niche_graph, fusion_items)

        if self.use_pnq:
            popular = popular - self._interest_embeddings(
                self.quadrant_niche_user_pop_item, fusion_items
            ) - self._interest_embeddings(self.quadrant_niche_user_niche_item, fusion_items)
            niche = niche - self._interest_embeddings(
                self.quadrant_pop_user_pop_item, fusion_items
            ) - self._interest_embeddings(self.quadrant_pop_user_niche_item, fusion_items)

        weights = F.softmax(self.interest_attention(torch.cat([popular, niche], dim=1)), dim=1)
        popular_weight, niche_weight = weights[:, :1], weights[:, 1:]
        visual_weighted = visual_embeddings * self.modal_gate(content)
        text_weighted = text_embeddings * self.modal_gate(content)
        popular_weighted = fusion_embeddings * self.modal_gate(popular) * popular_weight
        niche_weighted = fusion_embeddings * self.modal_gate(niche) * niche_weight
        if self.use_png:
            side = torch.stack(
                [visual_weighted, text_weighted, popular_weighted, niche_weighted], dim=0
            ).mean(dim=0)
        else:
            side = torch.stack([visual_embeddings, text_embeddings], dim=0).mean(dim=0)
        side = self.dropout(side)
        final = content + side
        users, items = torch.split(final, [self.n_users, self.n_items], dim=0)
        if train:
            return users, items, side, content
        return users, items

    @staticmethod
    def _info_nce(first, second, temperature):
        first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
        positive = torch.exp((first * second).sum(dim=1) / temperature)
        total = torch.exp(first @ second.T / temperature).sum(dim=1)
        return -torch.log(positive / total.clamp_min(1e-12)).mean()

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        user_all, item_all, side, content = self.forward(train=True)
        user_batch = user_all[users]
        positive_batch = item_all[positive]
        negative_batch = item_all[negative]
        recommendation = -F.logsigmoid(
            (user_batch * positive_batch).sum(dim=1) - (user_batch * negative_batch).sum(dim=1)
        ).mean()
        regularization = (
            user_batch.square().sum() + positive_batch.square().sum() + negative_batch.square().sum()
        ) / (2.0 * max(1, users.shape[0]))
        side_users, side_items = torch.split(side, [self.n_users, self.n_items], dim=0)
        content_users, content_items = torch.split(content, [self.n_users, self.n_items], dim=0)
        contrastive = self._info_nce(
            side_items[positive], content_items[positive], self.temperature
        ) + self._info_nce(side_users[users], content_users[users], self.temperature)
        loss = recommendation + self.reg_weight * regularization
        return loss + self.cl_weight * contrastive if self.use_pnl else loss

    def full_sort_predict(self, interaction):
        users, items = self.forward()
        return users[interaction[0]] @ items.T

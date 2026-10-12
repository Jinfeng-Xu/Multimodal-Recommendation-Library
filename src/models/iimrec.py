# coding: utf-8
"""One Graph, Multiple Gains: Single High-Quality Item-Item Graph (ACM MM 2026).

Adapted from https://github.com/Jinfeng-Xu/IIMRec for scalable MRLib execution.
"""

import math
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.graph_ops import (
    chunked_knn_graph,
    item_cooccurrence_graph,
    normalized_ui_graph,
    row_normalize,
)
from utils.graph_cache import GraphCacheManager


def _dct_ii(features):
    """Orthonormal DCT-II implemented with torch.fft (no extra dependency)."""
    length = features.shape[1]
    reordered = torch.cat([features[:, ::2], features[:, 1::2].flip(1)], dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = -torch.arange(length, device=features.device, dtype=features.dtype) * math.pi / (2 * length)
    transformed = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    transformed[:, 0] /= math.sqrt(length)
    if length > 1:
        transformed[:, 1:] /= math.sqrt(length / 2.0)
    return transformed


def _torch_sparse_to_scipy(graph):
    graph = graph.coalesce().cpu()
    row, column = graph.indices().numpy()
    return sp.csr_matrix((graph.values().numpy(), (row, column)), shape=graph.shape)


class IIMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if not hasattr(self, "visual_feat") or not hasattr(self, "textual_feat"):
            raise ValueError("IIMRec requires visual and textual item features")

        self.dimension = int(config["embedding_size"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.n_item_layers = int(config["n_mm_layers"])
        self.reg_weight = float(config["reg_weight"])
        self.dropout = float(config["dropout"])
        self.clu_weight = float(config["clu_lambda"])
        self.cli_weight = float(config["cli_lambda"])
        self.gpert_weight = float(config["gpert_lambda"])
        self.gpert_mask_p = float(config["gpert_mask_p"])
        self.temperature = float(config["temperature"])
        self.rig_on = bool(config["rig_on"])

        self.modalities = ["visual", "textual"]
        feature_map = {"visual": self.visual_feat, "textual": self.textual_feat}
        if hasattr(self, "audio_feat") and self.audio_feat is not None:
            self.modalities.append("audio")
            feature_map["audio"] = self.audio_feat
        self.feature_map = feature_map
        self.channels = len(self.modalities) + 1

        interaction = dataset.inter_matrix(form="coo").astype(np.float32)
        self.interaction = interaction
        norm_adj, _ = normalized_ui_graph(interaction, self.device)
        self.register_buffer("norm_adj", norm_adj)

        self.preferences = nn.ParameterDict()
        self.feature_mlps = nn.ModuleDict()
        self.user_projections = nn.ModuleDict()
        self.item_projections = nn.ModuleDict()
        for name in self.modalities:
            self.preferences[name] = nn.Parameter(torch.empty(self.n_users, self.dimension))
            nn.init.xavier_normal_(self.preferences[name])
            width = feature_map[name].shape[1]
            self.feature_mlps[name] = nn.Sequential(
                nn.Linear(width, 4 * self.dimension), nn.LeakyReLU(),
                nn.Linear(4 * self.dimension, self.dimension, bias=False)
            )
            self.user_projections[name] = nn.Linear(self.dimension, self.dimension, bias=False)
            self.item_projections[name] = nn.Linear(self.dimension, self.dimension, bias=False)

        spectral = torch.cat([_dct_ii(feature_map[name]) for name in self.modalities], dim=1)
        self.register_buffer("spectral_features", spectral)
        self.preferences["shared"] = nn.Parameter(torch.empty(self.n_users, self.dimension))
        nn.init.xavier_normal_(self.preferences["shared"])
        self.shared_mlp = nn.Sequential(
            nn.Linear(spectral.shape[1], 4 * self.dimension), nn.LeakyReLU(),
            nn.Linear(4 * self.dimension, self.dimension, bias=False)
        )
        self.user_projections["shared"] = nn.Linear(self.dimension, self.dimension, bias=False)
        self.item_projections["shared"] = nn.Linear(self.dimension, self.dimension, bias=False)

        cache = GraphCacheManager(config["data_path"], config["dataset"]).get_model_cache_dir("IIMRec")
        graph_name = (
            f"single_graph_k{int(config['knn_k'])}_co{int(config['item_co_topk'])}_"
            f"a{int(float(config['alpha_item_co']) * 100)}.pt"
        )
        graph_path = os.path.join(cache, graph_name)
        if os.path.exists(graph_path):
            item_graph = torch.load(graph_path, map_location=self.device).coalesce()
        else:
            item_graph = self._build_single_item_graph(config, interaction)
            torch.save(item_graph.cpu(), graph_path)
        self.register_buffer("item_graph", item_graph.to(self.device).coalesce())

        content_ui = self._build_content_guided_ui(interaction, self.item_graph, float(config["cui_gamma"]))
        self.register_buffer("content_ui", content_ui)
        self.masked_adj = self.content_ui

        representation_size = self.channels * self.dimension
        self.item_gate = nn.Sequential(
            nn.Linear(2 * representation_size, max(self.dimension, representation_size // 2)),
            nn.ReLU(), nn.Linear(max(self.dimension, representation_size // 2), 1), nn.Sigmoid()
        )
        self.result_embed = None

    def _build_single_item_graph(self, config, interaction):
        topk = int(config["knn_k"])
        chunk = int(config["knn_chunk_size"])
        modality_graphs = [
            chunked_knn_graph(self.feature_map[name], topk, chunk, binary=True)
            for name in self.modalities
        ]
        semantic = modality_graphs[0] * float(config["mm_image_weight"])
        remaining_weight = 1.0 - float(config["mm_image_weight"])
        for graph in modality_graphs[1:]:
            semantic = semantic + graph * (remaining_weight / max(1, len(modality_graphs) - 1))

        keep_ratio = float(config["dct_keep_ratio"])
        low_frequency = []
        for name in self.modalities:
            transformed = _dct_ii(self.feature_map[name])
            keep = max(1, int(transformed.shape[1] * keep_ratio))
            low_frequency.append(transformed[:, :keep])
        spectral_graph = chunked_knn_graph(torch.cat(low_frequency, dim=1), topk, chunk, binary=True)
        lambda_spec = float(config["lambda_spec"])
        semantic = row_normalize((1.0 - lambda_spec) * semantic + lambda_spec * spectral_graph)

        cooccurrence = item_cooccurrence_graph(
            interaction, int(config["item_co_topk"]), self.device
        )
        alpha = float(config["alpha_item_co"])
        return row_normalize(alpha * cooccurrence + (1.0 - alpha) * semantic)

    def _build_content_guided_ui(self, interaction, item_graph, gamma):
        expanded = (interaction.tocsr() @ _torch_sparse_to_scipy(item_graph)).tocsr()
        mixed = gamma * interaction.tocsr() + (1.0 - gamma) * expanded
        graph, _ = normalized_ui_graph(mixed.tocoo(), self.device)
        return graph

    def pre_epoch_processing(self):
        if self.dropout <= 0:
            self.masked_adj = self.content_ui
            return
        graph = self.content_ui.coalesce()
        keep = torch.rand(graph.values().shape, device=self.device) >= self.dropout
        self.masked_adj = torch.sparse_coo_tensor(
            graph.indices()[:, keep], graph.values()[keep] / (1.0 - self.dropout),
            graph.shape, device=self.device
        ).coalesce()

    def _entity_channels(self):
        channels = []
        for name in self.modalities:
            item = self.feature_mlps[name](self.feature_map[name])
            channels.append(torch.cat([self.preferences[name], item], dim=0))
        shared_item = self.shared_mlp(self.spectral_features)
        channels.append(torch.cat([self.preferences["shared"], shared_item], dim=0))
        return channels

    def forward(self):
        channels = self._entity_channels()
        embeddings = torch.cat([F.normalize(channel, dim=1) for channel in channels], dim=1)
        layers = [embeddings]
        for _ in range(self.n_ui_layers):
            embeddings = torch.sparse.mm(self.masked_adj, embeddings)
            layers.append(embeddings)
        propagated = torch.stack(layers, dim=0).sum(dim=0)
        users, items = torch.split(propagated, [self.n_users, self.n_items], dim=0)
        item_side = items
        for _ in range(self.n_item_layers):
            item_side = torch.sparse.mm(self.item_graph, item_side)
        if self.rig_on:
            items = items + self.item_gate(torch.cat([items, item_side], dim=1)) * item_side
        else:
            items = items + item_side
        self.result_embed = torch.cat([users, items], dim=0)
        return users, items, channels

    def _nce(self, first, second):
        first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
        labels = torch.arange(first.shape[0], device=first.device)
        return F.cross_entropy(first @ second.T / self.temperature, labels)

    def _channel_alignment(self, channels, users, items):
        total = torch.zeros((), device=self.device)
        pairs = list(zip(channels[:-1], channels[1:]))
        for first, second in pairs:
            total = total + self._nce(first[users], second[users])
            total = total + self._nce(first[self.n_users + items], second[self.n_users + items])
        return total / max(1, len(pairs))

    def _perturbation_loss(self, embeddings):
        keep = 1.0 - self.gpert_mask_p
        first = embeddings * (torch.rand_like(embeddings) > self.gpert_mask_p) / keep
        second = embeddings * (torch.rand_like(embeddings) > self.gpert_mask_p) / keep
        return self._nce(first, second)

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        user_all, item_all, channels = self.forward()
        user_batch, positive_batch, negative_batch = (
            user_all[users], item_all[positive], item_all[negative]
        )
        recommendation = -F.logsigmoid(
            (user_batch * positive_batch).sum(1) - (user_batch * negative_batch).sum(1)
        ).mean()
        regularization = self.reg_weight * (
            user_batch.square().mean() + positive_batch.square().mean() + negative_batch.square().mean()
        )
        unique_items = torch.unique(torch.cat([positive, negative]))
        alignment = self._channel_alignment(channels, users, unique_items)
        perturbed = self._perturbation_loss(
            torch.cat([user_batch, item_all[unique_items]], dim=0)
        )
        return recommendation + regularization + (self.clu_weight + self.cli_weight) * alignment + self.gpert_weight * perturbed

    def full_sort_predict(self, interaction):
        if self.result_embed is None:
            self.forward()
        users = self.result_embed[: self.n_users]
        items = self.result_embed[self.n_users :]
        return users[interaction[0]] @ items.T

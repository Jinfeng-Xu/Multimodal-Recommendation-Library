# coding: utf-8
"""Hierarchy-Aware Multimodal Distillation for Recommendation (TMM 2026).

Adapted from https://github.com/Tuo97/HSD at commit
9c0b2647a5d28eda95048cfcfa9ba98ed82a1520 for MRLib's data/trainer API.

The propagation, modality scaling, fused/channel BPR losses, and one-way
fusion-to-channel KL distillation follow the upstream implementation. The
standalone source stores four intended trainable embedding tensors as ordinary
attributes, which excludes them from ``Module.parameters()``. They are
registered as ``nn.Parameter`` here so the optimizer can update them.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class _Hyperboloid:
    """Minimal Lorentz-model operations used by HSD."""

    eps = {torch.float32: 1e-7, torch.float64: 1e-15}
    min_norm = 1e-15
    # cosh/sinh overflow far below 1e6 (about 89 for float32). Keeping the
    # Lorentz radius bounded also prevents exploding gradients in expmap0.
    max_theta = {torch.float32: 15.0, torch.float64: 30.0}

    @staticmethod
    def _arcosh(x):
        return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=1e-15)))

    def proj_tan0(self, u, curvature):
        result = u.clone()
        result[..., 0] = 0
        return result

    def proj(self, x, curvature):
        radius_sq = 1.0 / curvature
        spatial = x[..., 1:]
        result = x.clone()
        result[..., :1] = torch.sqrt(
            torch.clamp(radius_sq + spatial.square().sum(dim=-1, keepdim=True), min=self.eps[x.dtype])
        )
        return result

    def expmap0(self, u, curvature):
        radius = torch.sqrt(1.0 / curvature)
        spatial = u[..., 1:]
        norm = torch.clamp(torch.norm(spatial, p=2, dim=-1, keepdim=True), min=self.min_norm)
        theta = torch.clamp(norm / radius, max=self.max_theta.get(u.dtype, 15.0))
        result = torch.zeros_like(u)
        result[..., :1] = radius * torch.cosh(theta)
        result[..., 1:] = radius * torch.sinh(theta) * spatial / norm
        return self.proj(result, curvature)

    def logmap0(self, x, curvature):
        radius = torch.sqrt(1.0 / curvature)
        spatial = x[..., 1:]
        norm = torch.clamp(torch.norm(spatial, p=2, dim=-1, keepdim=True), min=self.min_norm)
        theta = torch.clamp(x[..., :1] / radius, min=1.0 + self.eps[x.dtype])
        result = torch.zeros_like(x)
        result[..., 1:] = radius * self._arcosh(theta) * spatial / norm
        return result


class _HyperbolicAggregator(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.register_buffer("graph", graph)

    def forward(self, embeddings):
        return torch.sparse.mm(self.graph, embeddings)


class HSD(GeneralRecommender):
    """MRLib-compatible HSD with dynamic visual/text feature dimensions."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        if self.visual_feat is None or self.textual_feat is None:
            raise ValueError("HSD requires both visual_feat and textual_feat")

        self.embedding_dim = int(config["embedding_size"]) + 1
        self.n_layers = int(config["n_layers"])
        self.channel_weight = float(config["channel_weight"])
        self.kl_weight = float(config["kl_weight"])
        self.tau = float(config["tau"])
        self.l2_weight = float(config["l2_weight"])

        self.manifold = _Hyperboloid()
        self.register_buffer("curvature", torch.tensor(float(config["curvature"])))

        # Register trainable tensors correctly; the official standalone code used
        # leaf tensors that were not exposed through Module.parameters().
        self.user_id_embedding = nn.Parameter(torch.empty(self.n_users, self.embedding_dim))
        self.item_id_embedding = nn.Parameter(torch.empty(self.n_items, self.embedding_dim))
        self.user_visual_embedding = nn.Parameter(torch.empty(self.n_users, self.embedding_dim))
        self.user_text_embedding = nn.Parameter(torch.empty(self.n_users, self.embedding_dim))
        for parameter in (
            self.user_id_embedding,
            self.item_id_embedding,
            self.user_visual_embedding,
            self.user_text_embedding,
        ):
            nn.init.xavier_uniform_(parameter)

        self.visual_embedding = nn.Embedding.from_pretrained(self.visual_feat, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(self.textual_feat, freeze=True)
        self.visual_projection = nn.Linear(self.visual_feat.shape[1], self.embedding_dim)
        self.text_projection = nn.Linear(self.textual_feat.shape[1], self.embedding_dim)

        interaction = dataset.inter_matrix(form="coo").astype(np.float32)
        adjacency = sp.bmat(
            [[None, interaction], [interaction.T, None]], format="coo", dtype=np.float32
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        inv_sqrt[degree == 0] = 0
        adjacency = (sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocoo()
        indices = torch.from_numpy(np.vstack([adjacency.row, adjacency.col]).astype(np.int64))
        values = torch.from_numpy(adjacency.data.astype(np.float32))
        graph = torch.sparse_coo_tensor(indices, values, adjacency.shape).coalesce().to(self.device)
        self.aggregators = nn.ModuleList(_HyperbolicAggregator(graph) for _ in range(self.n_layers))

    def _propagate(self, user_embeddings, item_embeddings):
        embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings = self.manifold.expmap0(
            self.manifold.proj_tan0(embeddings, self.curvature), self.curvature
        )
        layers = []
        for aggregator in self.aggregators:
            embeddings = aggregator(embeddings)
            layers.append(embeddings)
        return self.manifold.logmap0(torch.stack(layers, dim=1).mean(dim=1), self.curvature)

    def forward(self):
        visual_items = self.visual_projection(self.visual_embedding.weight)
        text_items = self.text_projection(self.text_embedding.weight)
        id_embeddings = self._propagate(self.user_id_embedding, self.item_id_embedding)
        visual_embeddings = 0.15 * self._propagate(self.user_visual_embedding, visual_items)
        text_embeddings = 0.75 * self._propagate(self.user_text_embedding, text_items)
        return id_embeddings, visual_embeddings, text_embeddings

    @staticmethod
    def _bpr(user, positive, negative):
        return -F.logsigmoid((user * positive).sum(dim=1) - (user * negative).sum(dim=1)).mean()

    def calculate_loss(self, interaction):
        users, positive, negative = interaction[:3]
        channels = self.forward()
        split_channels = [torch.split(x, [self.n_users, self.n_items], dim=0) for x in channels]

        positive_scores = []
        negative_scores = []
        channel_loss = 0.0
        for user_all, item_all in split_channels:
            user_batch = user_all[users]
            positive_batch = item_all[positive]
            negative_batch = item_all[negative]
            positive_scores.append((user_batch * positive_batch).sum(dim=1))
            negative_scores.append((user_batch * negative_batch).sum(dim=1))
            channel_loss = channel_loss + self._bpr(user_batch, positive_batch, negative_batch)

        fused_positive = torch.stack(positive_scores, dim=0).sum(dim=0)
        fused_negative = torch.stack(negative_scores, dim=0).sum(dim=0)
        recommendation_loss = -F.logsigmoid(fused_positive - fused_negative).mean()

        fused_distribution = F.softmax(
            torch.stack([fused_positive, fused_negative], dim=1) / self.tau, dim=1
        )
        distillation_loss = 0.0
        for pos_score, neg_score in zip(positive_scores, negative_scores):
            channel_distribution = F.softmax(
                torch.stack([pos_score, neg_score], dim=1) / self.tau, dim=1
            )
            distillation_loss = distillation_loss + F.kl_div(
                channel_distribution.clamp_min(1e-12).log(),
                fused_distribution,
                reduction="batchmean",
            )

        total_loss = (
            recommendation_loss
            + self.channel_weight * channel_loss
            + self.kl_weight * distillation_loss
        )
        if self.l2_weight > 0:
            regularization = sum(parameter.square().sum() for parameter in self.parameters())
            total_loss = total_loss + self.l2_weight * regularization
        return total_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        score = None
        for embeddings in self.forward():
            user_all, item_all = torch.split(embeddings, [self.n_users, self.n_items], dim=0)
            channel_score = user_all[users] @ item_all.T
            score = channel_score if score is None else score + channel_score
        return score

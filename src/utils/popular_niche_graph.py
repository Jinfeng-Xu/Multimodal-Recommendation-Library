# coding: utf-8
"""Scalable popular-niche graph construction for PNGRec."""

import numpy as np
import scipy.sparse as sp
import torch


class PopularNicheGraph:
    """Build PNGRec's wavelet-weighted interaction subgraphs in batches.

    The official implementation creates one dense signal per entity in a
    Python loop. This implementation applies the same DWT to CSR batches so
    MicroLens can be processed without allocating the full dense matrix.
    """

    def __init__(self, interaction, wavelet, level, device, batch_size=128):
        self.interaction = interaction.tocsr().astype(np.float32)
        self.device = device
        self.wavelet = wavelet
        self.level = int(level)
        self.batch_size = int(batch_size)

    def _base_statistics(self):
        coo = self.interaction.tocoo()
        rows = torch.from_numpy(coo.row.astype(np.int64)).to(self.device)
        cols = torch.from_numpy(coo.col.astype(np.int64)).to(self.device)
        values = torch.from_numpy(coo.data.astype(np.float32)).to(self.device)

        item_counts = torch.bincount(cols, minlength=self.interaction.shape[1]).float()
        item_popularity = torch.zeros(self.interaction.shape[1], device=self.device)
        item_popularity.scatter_add_(0, cols, values * torch.log1p(item_counts[cols]))
        item_popularity = item_popularity / item_popularity.norm().clamp_min(1e-8)

        user_activity = torch.zeros(self.interaction.shape[0], device=self.device)
        user_activity.scatter_add_(
            0, rows, values / torch.log1p(item_popularity[cols]).clamp_min(1e-8)
        )
        user_activity = user_activity / user_activity.norm().clamp_min(1e-8)
        return rows, cols, values, user_activity, item_popularity

    @torch.no_grad()
    def _wavelet_scores(self, matrix, scale):
        try:
            from pytorch_wavelets import DWT1D
        except ImportError as exc:
            raise ImportError("PNGRec requires pytorch-wavelets") from exc

        transform = DWT1D(wave=self.wavelet, J=self.level).to(self.device)
        low_scores, high_scores = [], []
        matrix = matrix.tocsr()
        for start in range(0, matrix.shape[0], self.batch_size):
            stop = min(start + self.batch_size, matrix.shape[0])
            signal = torch.from_numpy(matrix[start:stop].toarray()).to(self.device)
            signal = signal * (1.0 + scale[start:stop, None])
            low, highs = transform(signal.unsqueeze(1))
            low_scores.append(low.mean(dim=(1, 2)))
            high_scores.append(
                torch.stack([part.flatten(1).norm(dim=1) for part in highs], dim=1).sum(dim=1)
            )
        return torch.cat(low_scores), torch.cat(high_scores)

    @staticmethod
    def _dirichlet_weight(low, high):
        low = low.clamp_min(1e-6)
        high = high.clamp_min(1e-6)
        total = low + high
        alpha_low = (10.0 * low / total).cpu().numpy()
        alpha_high = (10.0 * high / total).cpu().numpy()
        left = np.random.gamma(alpha_low, 1.0).astype(np.float32)
        right = np.random.gamma(alpha_high, 1.0).astype(np.float32)
        denominator = left + right
        fallback = alpha_low / (alpha_low + alpha_high)
        weight = np.divide(left, denominator, out=fallback, where=denominator > 0)
        return torch.from_numpy(weight).to(low.device)

    @staticmethod
    def _sparse(rows, cols, values, shape):
        return torch.sparse_coo_tensor(
            torch.stack([rows, cols]), values, shape, device=values.device
        ).coalesce()

    @torch.no_grad()
    def build(self):
        rows, cols, values, user_activity, item_popularity = self._base_statistics()
        user_low, user_high = self._wavelet_scores(self.interaction, user_activity)
        item_low, item_high = self._wavelet_scores(self.interaction.T, item_popularity)
        user_pop = self._dirichlet_weight(user_low, user_high)
        item_pop = self._dirichlet_weight(item_low, item_high)
        user_niche, item_niche = 1.0 - user_pop, 1.0 - item_pop
        shape = self.interaction.shape

        quadrants = {
            "pop_user_pop_item": self._sparse(
                rows, cols, values * user_pop[rows] * item_pop[cols], shape
            ),
            "pop_user_niche_item": self._sparse(
                rows, cols, values * user_pop[rows] * item_niche[cols], shape
            ),
            "niche_user_pop_item": self._sparse(
                rows, cols, values * user_niche[rows] * item_pop[cols], shape
            ),
            "niche_user_niche_item": self._sparse(
                rows, cols, values * user_niche[rows] * item_niche[cols], shape
            ),
        }
        popular = quadrants["pop_user_pop_item"]
        niche = quadrants["niche_user_niche_item"]
        return popular, niche, quadrants

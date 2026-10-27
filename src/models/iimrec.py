

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

from common.abstract_recommender import GeneralRecommender
from types import SimpleNamespace

# Synced from Jinfeng-Xu/IIMRec commit b22a36911e3f41412de99102040b99cb0019872c.
# MRLib's Config exposes mapping access rather than attribute access, so optional
# official parameters use Config.get() to preserve the intended search grid.


class IIMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(IIMRec, self).__init__(config, dataset)

        # MRLib's modality auto-discovery uses descriptive attribute names;
        # retain the aliases expected by the official IIMRec implementation.
        self.v_feat = getattr(self, 'visual_feat', None)
        self.t_feat = getattr(self, 'textual_feat', None)
        self.a_feat = getattr(self, 'audio_feat', None)
        if self.v_feat is None or self.t_feat is None:
            raise ValueError("IIMRec requires both visual_feat and textual_feat")

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.decay_base = config['decay_base']
        self.decay_weight = config['decay_weight']
        self.cur_epoch = 0
        self.enable_residual = config['enable_residual']
        self.item_co_topk = config['item_co_topk']
        # Extract scalar value from list if needed
        alpha_co_val = config['alpha_item_co']
        self.alpha_item_co = alpha_co_val[0] if isinstance(alpha_co_val, list) else alpha_co_val
        self.n_nodes = self.n_users + self.n_items
        self.cli_lambda = config['cli_lambda']

        self.cui_topk = config['cui_topk']
        self.cui_theta = config['cui_theta']
        self.cui_gamma = config['cui_gamma']

        self.gpert_on = bool(config.get('gpert_on', True))
        self.gpert_lambda = config['gpert_lambda']
        self.gpert_tau = 0.2
        self.gpert_mask_p = config['gpert_mask_p']
        self.gpert_mode = 'mask'
        self.lambda_spec = float(config.get('lambda_spec', 0.1))

        self.dct_keep_ratio = float(config.get('dct_keep_ratio', 0.2))
        self.beta_lf = float(config.get('beta_lf', 0.5))
        self.beta_hf = float(config.get('beta_hf', 0.1))
        self.beta_mix = float(config.get('beta_mix', 0.4))
        self.spec_gate_tau = float(config.get('spec_gate_tau', 1))
        self.enable_spec_gate = bool(config.get('enable_spec_gate', True))

        self.ncer_alpha = float(config.get('ncer_alpha', 0.5))   # NCER boost strength

        # Residual II Gate
        self.rig_on = bool(config.get('rig_on', True))

        # II-Neighbor BPR Augmentation
        self.ina_lambda = float(config.get('ina_lambda', 0.0))   # augmented BPR weight
        self.ina_topk = int(config.get('ina_topk', 5))           # top-k II neighbors as soft positives
        self.ina_margin_discount = float(config.get('ina_margin_discount', 0.5))  # soft positive discount

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_norm_bipartite_from_ui(self.interaction_matrix.tocoo())
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        # Feature projections (same as DuGRec) - support for audio modality
        v_feat_dim, t_feat_dim, a_feat_dim = 0, 0, 0
        self.has_audio = self.a_feat is not None
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_feat_dim = self.v_feat.shape[1]
            self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim),
                dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.v_MLP = nn.Linear(v_feat_dim, 4 * self.embedding_dim)
            self.v_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_feat_dim = self.t_feat.shape[1]
            self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim),
                dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.t_MLP = nn.Linear(t_feat_dim, 4 * self.embedding_dim)
            self.t_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        # Audio modality support (for TikTok and similar datasets)
        if self.has_audio:
            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)
            a_feat_dim = self.a_feat.shape[1]
            self.a_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim),
                dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.a_MLP = nn.Linear(a_feat_dim, 4 * self.embedding_dim)
            self.a_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        self.id_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.n_users, self.embedding_dim),
            dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
        self.s_MLP = nn.Linear(t_feat_dim + v_feat_dim + a_feat_dim, 4 * self.embedding_dim)
        self.s_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        # DCT spectral features (same as DuGRec) - support audio if available
        if self.has_audio:
            w_t = dct.dct(self.t_feat, norm='ortho')
            w_v = dct.dct(self.v_feat, norm='ortho')
            w_a = dct.dct(self.a_feat, norm='ortho')
            self.interleaved_feat = torch.cat((w_v, w_t, w_a), 1)
        else:
            w_t = dct.dct(self.t_feat, norm='ortho')
            w_v = dct.dct(self.v_feat, norm='ortho')
            self.interleaved_feat = torch.cat((w_v, w_t), 1)

        # Build item graphs (with NCER enhancement)
        cache_dir = os.path.abspath(config['data_path'] + config['dataset'])
        self.build_and_cache_item_graphs(
            k_sem=self.knn_k,
            k_co=self.item_co_topk,
            alpha_item_co=self.alpha_item_co,
            cache_dir=cache_dir
        )

        # User CL setup (same as DuGRec) - support audio modality
        self.loss = SimpleNamespace(
            clu_tau=0.2,
            clu_lambda=config['clu_lambda'],
            pairs=set('vt,ts,va,as'.split(',') if self.has_audio else 'vt,ts'.split(',')),
        )

        self.u_proj_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.u_proj_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.u_proj_s = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        if self.has_audio:
            self.u_proj_a = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self._att_prior = self._build_modality_prior().to(self.device)
        self._att_ema = self._att_prior.clone()
        self._att_used = self._att_prior.clone()
        self._zu_v = self._zu_t = self._zu_s = None
        if self.has_audio:
            self._zu_a = None

        # Item CL setup (same as DuGRec) - support audio modality
        self.i_proj_v = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.i_proj_t = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.i_proj_s = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        if self.has_audio:
            self.i_proj_a = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.loss.cli_tau = 0.2
        self.loss.cli_lambda = self.cli_lambda
        self.loss.i_pairs = set('vt,ts,va,as'.split(',') if self.has_audio else 'vt,ts'.split(','))

        if self.rig_on:
            gate_dim = (4 if self.has_audio else 3) * self.embedding_dim
            self.ii_gate = nn.Sequential(
                nn.Linear(gate_dim * 2, gate_dim // 2),
                nn.ReLU(),
                nn.Linear(gate_dim // 2, 1),
                nn.Sigmoid()
            )

        self._ii_topk_neighbors = None  # will be set after mm_adj is built

        self.result_embed = None
        self._evaluation_cache_valid = False

        # Build content-guided UI graph
        self.adj_cui = self.build_cui_ui().to(self.device)

        # Build INA neighbor cache from mm_adj
        if self.ina_lambda > 0:
            self._build_ina_cache()

    def _build_ina_cache(self):
        with torch.no_grad():
            adj = self.mm_adj.coalesce()
            indices = adj.indices()   # [2, E]
            values = adj.values()     # [E]
            n = self.n_items
            k = self.ina_topk

            # For each item, find top-k neighbors by edge weight
            neighbors = torch.full((n, k), -1, dtype=torch.long, device=self.device)
            for i in range(n):
                mask = indices[0] == i
                if mask.sum() == 0:
                    continue
                cols = indices[1][mask]
                vals = values[mask]
                topk_count = min(k, cols.size(0))
                _, topk_idx = torch.topk(vals, topk_count)
                neighbors[i, :topk_count] = cols[topk_idx]

            self._ii_topk_neighbors = neighbors


    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
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
        return rows_inv_sqrt * cols_inv_sqrt

    def _row_normalize_sparse(self, indices, values, shape, eps=1e-12):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        adj = torch.sparse_coo_tensor(indices, values, size=shape, device=self.device).coalesce()
        rows = adj.indices()[0]
        vals = adj.values()
        row_sum = torch.zeros(shape[0], device=self.device, dtype=vals.dtype)
        row_sum.index_add_(0, rows, vals)
        norm_vals = vals / (row_sum[rows] + eps)
        return torch.sparse_coo_tensor(adj.indices(), norm_vals, size=shape, device=self.device).coalesce()

    def _row_softmax_sparse(self, indices, values, shape, eps=1e-12):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32, device=self.device)
        adj = torch.sparse_coo_tensor(indices, values, size=shape, device=self.device).coalesce()
        rows = adj.indices()[0]
        vals = adj.values()
        try:
            max_per_row = torch.full((shape[0],), -1e9, device=self.device, dtype=vals.dtype)
            max_per_row.scatter_reduce_(0, rows, vals, reduce='amax', include_self=True)
        except Exception:
            r_cpu = rows.detach().cpu().numpy()
            v_cpu = vals.detach().cpu().numpy()
            mp = np.full((shape[0],), -1e9, dtype=np.float32)
            for r, v in zip(r_cpu, v_cpu):
                if v > mp[r]:
                    mp[r] = v
            max_per_row = torch.tensor(mp, device=self.device, dtype=vals.dtype)
        e = torch.exp(vals - max_per_row[rows])
        denom = torch.zeros(shape[0], device=self.device, dtype=vals.dtype)
        denom.index_add_(0, rows, e)
        soft = e / (denom[rows] + eps)
        return torch.sparse_coo_tensor(adj.indices(), soft, size=shape, device=self.device).coalesce()

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def _split_dct_bands(self, W, keep_ratio):
        D = W.size(1)
        k = max(1, int(D * keep_ratio))
        return W[:, :k], W[:, k:]

    def _knn_from_feat(self, F_mat, k, shape):
        F_mat = F_mat / (F_mat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        sim = torch.mm(F_mat, F_mat.t())
        sim.fill_diagonal_(-1.0)
        _, nn_idx = torch.topk(sim, k, dim=-1)
        rows = torch.arange(nn_idx.size(0), device=F_mat.device).unsqueeze(1).expand(-1, k).reshape(-1)
        cols = nn_idx.reshape(-1)
        idx = torch.stack([rows, cols], 0)
        val = torch.ones(rows.numel(), device=F_mat.device, dtype=torch.float32)
        return self._row_normalize_sparse(idx, val, shape).coalesce()

    def build_item_semantic_graph(self, k=None, alpha_image=None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        parts = []
        if getattr(self, 'v_feat', None) is not None:
            emb = self.image_embedding.weight.detach()
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = torch.mm(emb, emb.t())
            sim.fill_diagonal_(-1.0)
            _, knn_ind = torch.topk(sim, k, dim=-1)
            rows = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = knn_ind.reshape(-1)
            indices = torch.stack([rows, cols], dim=0)
            values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)
            G_img = self._row_normalize_sparse(indices, values, (self.n_items, self.n_items))
            parts.append(G_img)
            del sim
        if getattr(self, 't_feat', None) is not None:
            emb = self.text_embedding.weight.detach()
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = torch.mm(emb, emb.t())
            sim.fill_diagonal_(-1.0)
            _, knn_ind = torch.topk(sim, k, dim=-1)
            rows = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = knn_ind.reshape(-1)
            indices = torch.stack([rows, cols], dim=0)
            values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)
            G_txt = self._row_normalize_sparse(indices, values, (self.n_items, self.n_items))
            parts.append(G_txt)
            del sim
        # Audio modality support
        if self.has_audio:
            emb = self.audio_embedding.weight.detach()
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-12)
            sim = torch.mm(emb, emb.t())
            sim.fill_diagonal_(-1.0)
            _, knn_ind = torch.topk(sim, k, dim=-1)
            rows = torch.arange(knn_ind.shape[0], device=self.device).unsqueeze(1).expand(-1, k).reshape(-1)
            cols = knn_ind.reshape(-1)
            indices = torch.stack([rows, cols], dim=0)
            values = torch.ones(rows.numel(), device=self.device, dtype=torch.float32)
            G_aud = self._row_normalize_sparse(indices, values, (self.n_items, self.n_items))
            parts.append(G_aud)
            del sim
        if len(parts) == 0:
            raise RuntimeError("No modality features.")
        
        # Handle different numbers of modalities
        if len(parts) == 1:
            G_sem = parts[0]
        elif len(parts) == 2:
            # Original bimodal (v, t)
            G_sem = alpha_image * parts[0] + (1.0 - alpha_image) * parts[1]
        elif len(parts) == 3:
            # Trimodal (v, t, a) - equal weighting or use alpha_image for v, rest split between t and a
            G_sem = alpha_image * parts[0] + (1.0 - alpha_image) * 0.5 * parts[1] + (1.0 - alpha_image) * 0.5 * parts[2]
        else:
            # Fallback: equal weighting
            G_sem = sum(parts) / len(parts)
        
        return G_sem.coalesce()

    def build_item_semantic_graph_spectral(self, k=None, alpha_image=None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        shape = (self.n_items, self.n_items)
        w_v = dct.dct(self.v_feat, norm='ortho')
        w_t = dct.dct(self.t_feat, norm='ortho')
        w_vt = self.interleaved_feat
        
        v_lf, v_hf = self._split_dct_bands(w_v, self.dct_keep_ratio)
        t_lf, t_hf = self._split_dct_bands(w_t, self.dct_keep_ratio)
        G_v_lf = self._knn_from_feat(v_lf, k, shape)
        G_v_hf = self._knn_from_feat(v_hf, k, shape)
        G_t_lf = self._knn_from_feat(t_lf, k, shape)
        G_t_hf = self._knn_from_feat(t_hf, k, shape)
        G_mix = self._knn_from_feat(w_vt, k, shape)

        # Handle audio modality if available
        if self.has_audio:
            w_a = dct.dct(self.a_feat, norm='ortho')
            a_lf, a_hf = self._split_dct_bands(w_a, self.dct_keep_ratio)
            G_a_lf = self._knn_from_feat(a_lf, k, shape)
            G_a_hf = self._knn_from_feat(a_hf, k, shape)
            
            # Trimodal fusion: v, t, a
            G_lf = alpha_image * G_v_lf + (1.0 - alpha_image) * 0.5 * G_t_lf + (1.0 - alpha_image) * 0.5 * G_a_lf
            G_hf = alpha_image * G_v_hf + (1.0 - alpha_image) * 0.5 * G_t_hf + (1.0 - alpha_image) * 0.5 * G_a_hf
        else:
            # Original bimodal fusion (v, t)
            G_lf = alpha_image * G_v_lf + (1.0 - alpha_image) * G_t_lf
            G_hf = alpha_image * G_v_hf + (1.0 - alpha_image) * G_t_hf
        
        G_spec = self.beta_lf * G_lf + self.beta_hf * G_hf + self.beta_mix * G_mix
        return G_spec.coalesce()

    def build_item_semantic_graph_joint(self, k=None, alpha_image=None):
        k = self.knn_k if k is None else int(k)
        alpha_image = self.mm_image_weight if alpha_image is None else float(alpha_image)
        G_spatial = self.build_item_semantic_graph(k=k, alpha_image=alpha_image)
        G_spec = self.build_item_semantic_graph_spectral(k=k, alpha_image=alpha_image)
        lam = float(self.lambda_spec)
        return ((1.0 - lam) * G_spatial + lam * G_spec).coalesce()

    def build_item_co_graph(self, topk=100):
        ui = self.interaction_matrix
        R = ui.tocsr()
        C = (R.T @ R).tocsr().astype(np.float32)
        C.setdiag(0.0)
        C.eliminate_zeros()
        rows, cols, vals = [], [], []
        n = C.shape[0]
        for i in range(n):
            start, end = C.indptr[i], C.indptr[i + 1]
            if start == end:
                continue
            row_idx = C.indices[start:end]
            row_val = C.data[start:end]
            if row_idx.size > topk:
                loc = np.argpartition(row_val, -topk)[-topk:]
                row_idx = row_idx[loc]
                row_val = row_val[loc]
            rows.extend([i] * row_idx.size)
            cols.extend(row_idx.tolist())
            vals.extend(row_val.tolist())
        if len(rows) == 0:
            indices = torch.empty((2, 0), dtype=torch.long, device=self.device)
            values = torch.empty((0,), dtype=torch.float32, device=self.device)
            return torch.sparse_coo_tensor(indices, values,
                                           size=(self.n_items, self.n_items), device=self.device)
        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.tensor(vals, dtype=torch.float32, device=self.device)
        return self._row_softmax_sparse(indices, values, (self.n_items, self.n_items)).coalesce()

    def _apply_ncer(self, G_fused):
        if self.ncer_alpha <= 0:
            return G_fused

        G = G_fused.coalesce()
        n = G.size(0)

        # Build binary adjacency for neighbor lookup
        ii = G.indices()
        binary_adj = torch.sparse_coo_tensor(
            ii, torch.ones(ii.size(1), device=self.device),
            size=(n, n), device=self.device
        ).coalesce()

        shared = torch.sparse.mm(binary_adj, binary_adj.t()).coalesce()

        # Look up shared counts for existing edges
        edges = G.indices()  # [2, E]
        vals = G.values()    # [E]

        # Create index for lookup: convert (row, col) to linear index
        edge_linear = edges[0] * n + edges[1]
        shared_indices = shared.indices()
        shared_linear = shared_indices[0] * n + shared_indices[1]
        shared_vals = shared.values()

        # Exact sparse lookup for the common-neighbor count of each fused edge.
        # The official dense E-by-N gather can require several GB even on Baby;
        # coalesced sparse indices are lexicographically sorted, so searchsorted
        # preserves the same values with O(E + nnz(shared)) memory.
        shared_count = torch.zeros(edges.size(1), device=self.device, dtype=shared_vals.dtype)
        if shared_linear.numel() > 0:
            positions = torch.searchsorted(shared_linear, edge_linear)
            in_range = positions < shared_linear.numel()
            safe_positions = positions.clamp(max=shared_linear.numel() - 1)
            matched = in_range & (shared_linear[safe_positions] == edge_linear)
            shared_count[matched] = shared_vals[safe_positions[matched]]

        # Normalize by k (max possible shared neighbors)
        k = float(self.knn_k)
        consistency = shared_count / (k + 1e-12)
        consistency = consistency.clamp(0, 1)

        # Boost: edges with high consistency get higher weight
        boost = 1.0 + self.ncer_alpha * consistency
        new_vals = vals * boost

        G_out = torch.sparse_coo_tensor(edges, new_vals, size=G.size(), device=self.device)
        # Re-normalize rows
        return self._row_normalize_sparse(
            G_out.coalesce().indices(),
            G_out.coalesce().values(),
            G_out.size()
        ).coalesce()


    def build_and_cache_item_graphs(self, k_sem, k_co, alpha_item_co, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        # Encode every graph-building option in the cache key. The v2 prefix
        # prevents graphs created by older, incomplete keys from being reused.
        audio_suffix = '_a' if self.has_audio else ''
        fused_path = os.path.join(
            cache_dir,
            f'NERec_Gii_v2_ksem{k_sem}_kco{k_co}_aco{alpha_item_co:g}'
            f'_aimg{self.mm_image_weight:g}_lspec{self.lambda_spec:g}'
            f'_ncer{self.ncer_alpha:g}_dct{self.dct_keep_ratio:g}'
            f'_blf{self.beta_lf:g}_bhf{self.beta_hf:g}_bmix{self.beta_mix:g}'
            f'_gate{int(self.enable_spec_gate)}_gtau{self.spec_gate_tau:g}'
            f'{audio_suffix}.pt'
        )
        if os.path.exists(fused_path):
            self.mm_adj = torch.load(fused_path, map_location=self.device).coalesce().to(self.device)
            return

        # Same fusion as DuGRec
        G_sem = self.build_item_semantic_graph_joint(k=k_sem, alpha_image=self.mm_image_weight)
        G_co = self.build_item_co_graph(topk=k_co)

        if self.enable_spec_gate:
            with torch.no_grad():
                # Support audio modality in spectral gate
                if self.has_audio:
                    W = torch.cat([dct.dct(self.v_feat, norm='ortho'),
                                   dct.dct(self.t_feat, norm='ortho'),
                                   dct.dct(self.a_feat, norm='ortho')], dim=1)
                else:
                    W = torch.cat([dct.dct(self.v_feat, norm='ortho'),
                                   dct.dct(self.t_feat, norm='ortho')], dim=1)
                lf, _ = self._split_dct_bands(W, self.dct_keep_ratio)
                rho = lf.norm(p=2, dim=1) / (W.norm(p=2, dim=1) + 1e-12)
            G_co = G_co.coalesce()
            ii = G_co.indices()
            vv = G_co.values()
            gate = torch.exp(-(rho[ii[0]] - rho[ii[1]]).abs() / max(1e-6, self.spec_gate_tau))
            G_co = torch.sparse_coo_tensor(ii, vv * gate, size=G_co.shape, device=G_co.device).coalesce()

        G_fused = (alpha_item_co * G_co + (1.0 - alpha_item_co) * G_sem).coalesce()

        G_fused = self._apply_ncer(G_fused)

        self.mm_adj = G_fused.to(self.device)
        torch.save(self.mm_adj.cpu(), fused_path)


    def build_norm_bipartite_from_ui(self, ui_coo):
        n_u, n_i = self.n_users, self.n_items
        ui = ui_coo.tocoo()
        Rt = ui.transpose().tocoo()
        A = sp.dok_matrix((n_u + n_i, n_u + n_i), dtype=np.float32)
        data_dict = dict(zip(zip(ui.row, ui.col + n_u), [1] * ui.nnz))
        data_dict.update(dict(zip(zip(Rt.row + n_u, Rt.col), [1] * Rt.nnz)))
        A._update(data_dict)
        A = A.tocsr()
        deg = np.asarray((A > 0).sum(axis=1)).ravel().astype(np.float32) + 1e-7
        inv_sqrt = np.power(deg, -0.5)
        D = sp.diags(inv_sqrt)
        L = (D @ A @ D).tocoo()
        idx = torch.LongTensor(np.vstack([L.row, L.col]))
        val = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(idx, val, torch.Size((n_u + n_i, n_u + n_i))).to(self.device)

    def torch_sparse_to_scipy_csr(self, T):
        T = T.coalesce().cpu()
        row, col = T.indices().numpy()
        data = T.values().numpy()
        return sp.csr_matrix((data, (row, col)), shape=T.shape)

    def _filter_sparse_topk_threshold(self, S, topk, theta):
        S = S.coalesce()
        rows, cols, vals = S.indices()[0], S.indices()[1], S.values()
        n_rows = S.size(0)
        buckets = [[] for _ in range(n_rows)]
        for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
            if v >= theta:
                buckets[r].append((c, v))
        f_rows, f_cols, f_vals = [], [], []
        for r, lst in enumerate(buckets):
            if not lst:
                continue
            if len(lst) > topk:
                lst = sorted(lst, key=lambda x: x[1], reverse=True)[:topk]
            for c, v in lst:
                f_rows.append(r)
                f_cols.append(c)
                f_vals.append(v)
        if not f_rows:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.float32, device=self.device),
                size=S.shape, device=self.device)
        idx = torch.tensor([f_rows, f_cols], dtype=torch.long, device=self.device)
        val = torch.tensor(f_vals, dtype=torch.float32, device=self.device)
        return self._row_normalize_sparse(idx, val, S.shape).coalesce()

    def build_cui_ui(self):
        S = self.mm_adj
        S_f = self._filter_sparse_topk_threshold(S, topk=self.cui_topk, theta=self.cui_theta)
        R = self.interaction_matrix.tocsr()
        S_csr = self.torch_sparse_to_scipy_csr(S_f)
        R_tilde = (R @ S_csr).tocsr()
        if self.cui_gamma is not None:
            R_mix = self.cui_gamma * R + (1.0 - self.cui_gamma) * R_tilde
        else:
            R_mix = R_tilde
        return self.build_norm_bipartite_from_ui(R_mix.tocoo())

    def _apply_dropedge(self, adj):
        if self.dropout <= 0.0:
            return adj
        adj = adj.coalesce()
        idx = adj.indices()
        val = adj.values()
        m = int(val.size(0) * (1. - self.dropout))
        prob = (val / (val.sum() + 1e-12)).clamp_min(1e-12)
        keep = torch.multinomial(prob, m, replacement=False)
        k_idx = idx[:, keep]
        k_val = val[keep]
        return torch.sparse_coo_tensor(k_idx, k_val, size=adj.shape, device=adj.device).coalesce()

    def pre_epoch_processing(self):
        self.cur_epoch += 1
        if getattr(self, 'adj_cui', None) is None:
            raise RuntimeError("adj_cui is None")
        self.masked_adj = self._apply_dropedge(self.adj_cui)

    def _lightgcn_propagate(self, adj, v_x):
        ego = v_x
        all_embeddings = [ego]
        if self.enable_residual:
            for _ in range(self.n_ui_layers):
                side = torch.sparse.mm(adj, ego)
                _w = F.cosine_similarity(side, ego, dim=-1)
                side = torch.einsum('a,ab->ab', _w, side)
                ego = side
                all_embeddings.append(ego)
        else:
            for _ in range(self.n_ui_layers):
                ego = torch.sparse.mm(adj, ego)
                all_embeddings.append(ego)
        H = torch.stack(all_embeddings, dim=0).sum(dim=0)
        item_rep = H[self.n_users:]
        uvw = H[:self.n_users]
        
        # Split user representations based on number of modalities
        d = self.embedding_dim
        v_rep = uvw[:, :d]
        t_rep = uvw[:, d:2 * d]
        if self.has_audio:
            a_rep = uvw[:, 2 * d:3 * d]
            s_rep = uvw[:, 3 * d:]
            user_rep = self._cl_fuse_users(v_rep, t_rep, s_rep, a_rep)
        else:
            s_rep = uvw[:, 2 * d:]
            user_rep = self._cl_fuse_users(v_rep, t_rep, s_rep)
        
        return user_rep, item_rep

    def _cl_fuse_users(self, v_rep, t_rep, s_rep, a_rep=None):
        z_v = F.normalize(self.u_proj_v(v_rep), dim=1)
        z_t = F.normalize(self.u_proj_t(t_rep), dim=1)
        z_s = F.normalize(self.u_proj_s(s_rep), dim=1)
        
        if self.has_audio and a_rep is not None:
            z_a = F.normalize(self.u_proj_a(a_rep), dim=1)
            # 4-modality attention scores
            sc_v = (z_v * z_t).sum(1, keepdim=True) + (z_v * z_s).sum(1, keepdim=True) + (z_v * z_a).sum(1, keepdim=True)
            sc_t = (z_t * z_v).sum(1, keepdim=True) + (z_t * z_s).sum(1, keepdim=True) + (z_t * z_a).sum(1, keepdim=True)
            sc_s = (z_s * z_v).sum(1, keepdim=True) + (z_s * z_t).sum(1, keepdim=True) + (z_s * z_a).sum(1, keepdim=True)
            sc_a = (z_a * z_v).sum(1, keepdim=True) + (z_a * z_t).sum(1, keepdim=True) + (z_a * z_s).sum(1, keepdim=True)
            att_new = F.softmax(torch.cat([sc_v, sc_t, sc_s, sc_a], 1) / self.loss.clu_tau, dim=1)
            fused = torch.cat([
                self._att_used[:, [0]] * v_rep,
                self._att_used[:, [1]] * t_rep,
                self._att_used[:, [2]] * s_rep,
                self._att_used[:, [3]] * a_rep
            ], dim=1)
            self._zu_v, self._zu_t, self._zu_s, self._zu_a = z_v, z_t, z_s, z_a
        else:
            # Original 3-modality attention (v, t, s) - keep unchanged for baby dataset
            sc_v = (z_v * z_t).sum(1, keepdim=True) + (z_v * z_s).sum(1, keepdim=True)
            sc_t = (z_t * z_v).sum(1, keepdim=True) + (z_t * z_s).sum(1, keepdim=True)
            sc_s = (z_s * z_v).sum(1, keepdim=True) + (z_s * z_t).sum(1, keepdim=True)
            att_new = F.softmax(torch.cat([sc_v, sc_t, sc_s], 1) / self.loss.clu_tau, dim=1)
            fused = torch.cat([
                self._att_used[:, [0]] * v_rep,
                self._att_used[:, [1]] * t_rep,
                self._att_used[:, [2]] * s_rep
            ], dim=1)
            self._zu_v, self._zu_t, self._zu_s = z_v, z_t, z_s

        if self.training:
            beta_e = self.decay_weight * (1.0 - (self.decay_base ** max(1, self.cur_epoch)))
            beta_e = float(max(0.0, min(1.0, beta_e)))
            self._att_ema = (1.0 - beta_e) * self._att_ema + beta_e * att_new.detach()
            self._att_used = self._att_ema

        return fused

    def _build_modality_prior(self):
        ui = self.interaction_matrix.tocsr().astype(np.float32)
        deg = np.maximum(ui.getnnz(axis=1), 1).astype(np.float32)
        v_norm = np.linalg.norm(self.v_feat.cpu().numpy(), axis=1) if self.v_feat is not None else None
        t_norm = np.linalg.norm(self.t_feat.cpu().numpy(), axis=1) if self.t_feat is not None else None
        a_norm = np.linalg.norm(self.a_feat.cpu().numpy(), axis=1) if self.has_audio else None
        nU = self.n_users
        sv = np.ones(nU, dtype=np.float32)
        st = np.ones(nU, dtype=np.float32)
        sa = np.ones(nU, dtype=np.float32) if self.has_audio else None
        if v_norm is not None:
            sv = (ui @ v_norm) / deg
        if t_norm is not None:
            st = (ui @ t_norm) / deg
        if self.has_audio and a_norm is not None:
            sa = (ui @ a_norm) / deg
        
        if self.has_audio:
            ss = (sv + st + sa) / 3.0
            prior = torch.tensor(np.stack([sv, st, ss, sa], axis=1), dtype=torch.float32)
        else:
            ss = 0.5 * (sv + st)
            prior = torch.tensor(np.stack([sv, st, ss], axis=1), dtype=torch.float32)
        return torch.softmax(prior / 1.0, dim=1)

    def forward(self, _unused_adj=None):
        # Process visual and text features (always present)
        tmp_v_feat = self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat)))
        tmp_t_feat = self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat)))
        tmp_s_feat = self.s_MLP_1(F.leaky_relu(self.s_MLP(self.interleaved_feat)))
        
        # Audio modality if available
        if self.has_audio:
            tmp_a_feat = self.a_MLP_1(F.leaky_relu(self.a_MLP(self.a_feat)))
            rep_ua = torch.cat((self.a_preference, tmp_a_feat), dim=0)
            rep_sh = torch.cat((self.id_preference, tmp_s_feat), dim=0)
        else:
            rep_ua = None
            rep_sh = torch.cat((self.id_preference, tmp_s_feat), dim=0)

        rep_uv = torch.cat((self.v_preference, tmp_v_feat), dim=0)
        rep_ut = torch.cat((self.t_preference, tmp_t_feat), dim=0)
        
        # Concatenate user representations: [v_rep, t_rep, s_rep] or [v_rep, t_rep, a_rep, s_rep]
        if self.has_audio:
            v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_ua), F.normalize(rep_sh)), 1)
        else:
            v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_sh)), 1)

        user_rep, item_rep = self._lightgcn_propagate(self.masked_adj, v_x)

        d = self.embedding_dim
        v_i = item_rep[:, :d]
        t_i = item_rep[:, d:2 * d]
        if self.has_audio:
            a_i = item_rep[:, 2 * d:3 * d]
            s_i = item_rep[:, 3 * d:]
        else:
            s_i = item_rep[:, 2 * d:]

        self._zi_iv = F.normalize(self.i_proj_v(v_i), dim=1)
        self._zi_it = F.normalize(self.i_proj_t(t_i), dim=1)
        self._zi_is = F.normalize(self.i_proj_s(s_i), dim=1)
        
        if self.has_audio:
            self._zi_ia = F.normalize(self.i_proj_a(a_i), dim=1)

        # II propagation (same as DuGRec)
        h = item_rep
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # [Innovation 2a] Residual II Gate (RIG)
        if self.rig_on and hasattr(self, 'ii_gate'):
            gate_input = torch.cat([item_rep, h], dim=1)
            gate = self.ii_gate(gate_input)
            item_rep = item_rep + gate * h
        else:
            item_rep = item_rep + h

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        return user_rep, item_rep

    def _nce(self, A, B, tau):
        logits = (A @ B.t()) / tau
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    # User CL (same as DuGRec) - support audio
    def _user_cl_total(self, users):
        device = users.device
        total = torch.tensor(0.0, device=device)
        stats = {}
        if self.loss.clu_lambda > 0:
            zv = self._zu_v[users]
            zt = self._zu_t[users]
            zs = self._zu_s[users]
            cl = torch.tensor(0.0, device=device)
            
            if self.has_audio:
                za = self._zu_a[users]
                # 4-modality contrastive learning
                if 'vt' in self.loss.pairs:
                    cl = cl + self._nce(zv, zt, self.loss.clu_tau)
                if 'ts' in self.loss.pairs:
                    cl = cl + self._nce(zt, zs, self.loss.clu_tau)
                if 'va' in self.loss.pairs:
                    cl = cl + self._nce(zv, za, self.loss.clu_tau)
                if 'as' in self.loss.pairs:
                    cl = cl + self._nce(za, zs, self.loss.clu_tau)
            else:
                # Original 3-modality CL for baby dataset
                if 'vt' in self.loss.pairs:
                    cl = cl + self._nce(zv, zt, self.loss.clu_tau)
                if 'ts' in self.loss.pairs:
                    cl = cl + self._nce(zt, zs, self.loss.clu_tau)
            
            total += self.loss.clu_lambda * cl
            stats['cl_u'] = float(cl.item())
        return total, stats

    # Item CL (same as DuGRec) - support audio
    def _item_mod_cl_total(self, items_unique):
        device = items_unique.device
        total = torch.tensor(0.0, device=device)
        stats = {}
        if getattr(self.loss, 'cli_lambda', 0.0) <= 0:
            return total, stats
        z_v_all = getattr(self, '_zi_iv', None)
        z_t_all = getattr(self, '_zi_it', None)
        z_s_all = getattr(self, '_zi_is', None)
        z_a_all = getattr(self, '_zi_ia', None) if self.has_audio else None
        
        if z_v_all is None or z_t_all is None or z_s_all is None:
            return total, stats
        
        zv = z_v_all[items_unique]
        zt = z_t_all[items_unique]
        zs = z_s_all[items_unique]
        cl = torch.tensor(0.0, device=device)
        pairs = getattr(self.loss, 'i_pairs', set(('vt', 'ts')))
        
        if self.has_audio and z_a_all is not None:
            za = z_a_all[items_unique]
            # 4-modality item CL
            if 'vt' in pairs:
                cl = cl + self._nce(zv, zt, self.loss.cli_tau)
            if 'ts' in pairs:
                cl = cl + self._nce(zt, zs, self.loss.cli_tau)
            if 'va' in pairs:
                cl = cl + self._nce(zv, za, self.loss.cli_tau)
            if 'as' in pairs:
                cl = cl + self._nce(za, zs, self.loss.cli_tau)
        else:
            # Original 3-modality item CL for baby dataset
            if 'vt' in pairs:
                cl = cl + self._nce(zv, zt, self.loss.cli_tau)
            if 'ts' in pairs:
                cl = cl + self._nce(zt, zs, self.loss.cli_tau)
        
        total = total + self.loss.cli_lambda * cl
        stats['cl_i_mod'] = float(cl.item())
        return total, stats

    def _ina_augmented_bpr(self, users_emb, pos_items, ia_embeddings):
        device = pos_items.device
        if self.ina_lambda <= 0 or self._ii_topk_neighbors is None:
            return torch.tensor(0.0, device=device), {}

        # Get II-graph neighbors of positive items
        neighbors = self._ii_topk_neighbors[pos_items]  # [B, topk]

        # Filter out -1 (padding for items with fewer than topk neighbors)
        valid_mask = neighbors >= 0  # [B, topk]
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device), {}

        # Pick one random valid neighbor per user as soft positive
        # (more efficient than using all topk)
        rand_idx = torch.zeros(pos_items.size(0), dtype=torch.long, device=device)
        for b in range(pos_items.size(0)):
            valid_count = valid_mask[b].sum()
            if valid_count > 0:
                rand_idx[b] = torch.randint(0, valid_count.item(), (1,), device=device)
            # if no valid neighbors, rand_idx stays 0, will be masked out

        # Gather the chosen neighbor
        valid_neighbors_flat = neighbors.gather(1, rand_idx.unsqueeze(1)).squeeze(1)  # [B]
        has_neighbor = valid_mask.any(dim=1)  # [B] bool

        if has_neighbor.sum() == 0:
            return torch.tensor(0.0, device=device), {}

        # Soft positive embeddings
        soft_pos_emb = ia_embeddings[valid_neighbors_flat]  # [B, D]

        # Random negatives (same as standard BPR)
        neg_items = torch.randint(0, self.n_items, (pos_items.size(0),), device=device)
        neg_emb = ia_embeddings[neg_items]

        # Discounted BPR: soft_positive vs negative, weighted down
        pos_scores = (users_emb * soft_pos_emb).sum(dim=1)
        neg_scores = (users_emb * neg_emb).sum(dim=1)
        margin = F.logsigmoid(pos_scores - neg_scores)

        # Only count items that have valid neighbors
        loss = -self.ina_margin_discount * (margin * has_neighbor.float()).sum() / (has_neighbor.sum() + 1e-8)
        return self.ina_lambda * loss, {'ina_bpr': float(loss.item())}

    # Graph perturbation SSL (standard uniform masking, same as DuGRec)
    def _graph_perturb_ssl(self, users, batch_items, ua_embeddings, ia_embeddings):
        if not getattr(self, 'gpert_on', False) or self.gpert_lambda <= 0:
            return torch.tensor(0., device=ua_embeddings.device), {}

        def _mk_views(x, p):
            m1 = (torch.rand_like(x) > p).float()
            m2 = (torch.rand_like(x) > p).float()
            keep = 1.0 - p
            e1 = F.normalize(x * m1 / keep, dim=1)
            e2 = F.normalize(x * m2 / keep, dim=1)
            return e1, e2

        p = self.gpert_mask_p
        u1, u2 = _mk_views(ua_embeddings[users], p)
        i1, i2 = _mk_views(ia_embeddings[batch_items], p)

        ssl_u = self._nce(u1, u2, self.gpert_tau)
        ssl_i = self._nce(i1, i2, self.gpert_tau)
        reg = self.gpert_lambda * (ssl_u + ssl_i)
        return reg, {'gpert_u': float(ssl_u.item()), 'gpert_i': float(ssl_i.item())}


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        return -torch.mean(maxi)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g = ua_embeddings[users]
        pos_i_g = ia_embeddings[pos_items]
        neg_i_g = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g, pos_i_g, neg_i_g)
        reg_loss = self.reg_weight * ((ua_embeddings ** 2).mean() + (ia_embeddings ** 2).mean())

        # User CL (same as DuGRec)
        u_reg, _ = self._user_cl_total(users)

        # Item modality CL (same as DuGRec)
        batch_items = torch.unique(torch.cat([pos_items, neg_items], dim=0))
        i_mod_reg, _ = self._item_mod_cl_total(batch_items)

        # Graph perturbation CL (same as DuGRec, no MACA)
        gpert_reg, _ = self._graph_perturb_ssl(users, batch_items, ua_embeddings, ia_embeddings)

        ina_reg, _ = self._ina_augmented_bpr(u_g, pos_items, ia_embeddings)

        # The optimizer updates parameters after this method returns, so the
        # embeddings computed above must not be reused for evaluation.
        self._evaluation_cache_valid = False
        return batch_mf_loss + reg_loss + u_reg + i_mod_reg + gpert_reg + ina_reg

    def full_sort_predict(self, interaction):
        if self.result_embed is None or not self._evaluation_cache_valid:
            self.forward(self.masked_adj)
            self._evaluation_cache_valid = True
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

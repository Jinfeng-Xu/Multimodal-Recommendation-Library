# coding: utf-8
"""Sparse graph helpers shared by scalable model adaptations."""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def scipy_to_torch(matrix, device):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce().to(device)


def normalized_ui_graph(interaction, device):
    interaction = interaction.tocoo().astype(np.float32)
    adjacency = sp.bmat(
        [[None, interaction], [interaction.T, None]], format="coo", dtype=np.float32
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    normalized = (sp.diags(inv_sqrt) @ adjacency @ sp.diags(inv_sqrt)).tocsr()
    return scipy_to_torch(normalized, device), scipy_to_torch(
        normalized[: interaction.shape[0], interaction.shape[0] :], device
    )


def row_normalize(graph):
    graph = graph.coalesce()
    indices, values = graph.indices(), graph.values()
    degree = torch.zeros(graph.shape[0], dtype=values.dtype, device=values.device)
    degree.scatter_add_(0, indices[0], values)
    values = values / degree[indices[0]].clamp_min(1e-12)
    return torch.sparse_coo_tensor(indices, values, graph.shape, device=values.device).coalesce()


@torch.no_grad()
def chunked_knn_graph(features, topk, chunk_size=1024, binary=False):
    """Cosine KNN without allocating the full item-item similarity matrix."""
    features = F.normalize(features, dim=1)
    n_items = features.shape[0]
    k = min(int(topk), n_items)
    rows, columns, weights = [], [], []
    for start in range(0, n_items, int(chunk_size)):
        stop = min(start + int(chunk_size), n_items)
        similarity = features[start:stop] @ features.T
        local_rows = torch.arange(start, stop, device=features.device)
        similarity[torch.arange(stop - start, device=features.device), local_rows] = -1.0
        values, indices = torch.topk(similarity, k=k, dim=1)
        rows.append(local_rows[:, None].expand_as(indices).reshape(-1))
        columns.append(indices.reshape(-1))
        weights.append(torch.ones_like(values).reshape(-1) if binary else values.reshape(-1).clamp_min(1e-7))
    graph = torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows), torch.cat(columns)]),
        torch.cat(weights),
        (n_items, n_items),
        device=features.device,
    ).coalesce()
    return row_normalize(graph)


@torch.no_grad()
def item_cooccurrence_graph(interaction, topk, device):
    """Sparse top-k item co-occurrence graph."""
    cooccurrence = (interaction.tocsr().T @ interaction.tocsr()).tocsr().astype(np.float32)
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    rows, columns, values = [], [], []
    for row in range(cooccurrence.shape[0]):
        start, stop = cooccurrence.indptr[row], cooccurrence.indptr[row + 1]
        column = cooccurrence.indices[start:stop]
        value = cooccurrence.data[start:stop]
        if value.size > topk:
            keep = np.argpartition(value, -topk)[-topk:]
            column, value = column[keep], value[keep]
        rows.extend([row] * len(column))
        columns.extend(column.tolist())
        values.extend(value.tolist())
    if not rows:
        index = torch.empty((2, 0), dtype=torch.long, device=device)
        value = torch.empty(0, dtype=torch.float32, device=device)
    else:
        index = torch.tensor([rows, columns], dtype=torch.long, device=device)
        value = torch.tensor(values, dtype=torch.float32, device=device)
    return row_normalize(
        torch.sparse_coo_tensor(index, value, cooccurrence.shape, device=device).coalesce()
    )

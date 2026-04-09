import torch


def samples_gen(user_rep, item_rep, k):
    # 假设 user_rep 和 item_rep 已经定义并且是正确的维度
    # user_rep: (A, 128), item_rep: (B, 128)

    # 步骤1: 标准化向量
    user_rep = user_rep / user_rep.norm(dim=1, keepdim=True)
    item_rep = item_rep / item_rep.norm(dim=1, keepdim=True)

    # 步骤2: compute余弦相似度矩阵
    pos_similarity_matrix = torch.mm(user_rep, item_rep.t())  # 结果维度为 (A, B)
    neg_similarity_matrix = -pos_similarity_matrix
    # 步骤3: 对每个用户找到最相似的k个项目的位置
    _, top_k_pos_indices = torch.topk(pos_similarity_matrix, k=k, dim=1)
    _, top_k_neg_indices = torch.topk(neg_similarity_matrix, k=k, dim=1)

    # top_k_indices 就是每个用户最相似的k个项目的位置，维度为 (A, k)
    return top_k_pos_indices, top_k_neg_indices
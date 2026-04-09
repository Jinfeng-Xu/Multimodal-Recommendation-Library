# coding: utf-8
"""
图结构缓存工具
用于build和缓存各种模型需要的图结构数据

支持的缓存类型：
1. DualGNN: user-user 图（基于 KNN）
2. FREEDOM: item-item KNN 图
3. 其他模型的图结构
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import pickle
from logging import getLogger


class GraphCacheManager:
    """Graph caches管理器"""
    
    def __init__(self, data_path: str, dataset_name: str, cache_dir: str = 'cache'):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.cache_dir = os.path.join(data_path, dataset_name, cache_dir)
        self.logger = getLogger()
        
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.logger.info(f'Graph cache directory: {self.cache_dir}')
    
    def get_model_cache_dir(self, model_name: str) -> str:
        """获取特定模型的缓存目录"""
        model_cache_dir = os.path.join(self.cache_dir, model_name)
        if not os.path.exists(model_cache_dir):
            os.makedirs(model_cache_dir)
        return model_cache_dir
    
    def clear_cache(self, model_name: Optional[str] = None):
        """清除缓存
        
        Args:
            model_name: 如果指定，只清除该模型的缓存；否则清除所有
        """
        if model_name:
            model_cache_dir = self.get_model_cache_dir(model_name)
            if os.path.exists(model_cache_dir):
                import shutil
                shutil.rmtree(model_cache_dir)
                self.logger.info(f'Cleared cache for model: {model_name}')
        else:
            if os.path.exists(self.cache_dir):
                import shutil
                shutil.rmtree(self.cache_dir)
                self.logger.info('Cleared all cache')
    
    def save_graph(self, model_name: str, graph_name: str, graph_data, metadata: Dict = None):
        """save图结构到缓存
        
        Args:
            model_name: Model name
            graph_name: 图名称
            graph_data: 图数据（torch.Tensor, scipy.sparse, dict 等）
            metadata: 元数据（configurationparameter等）
        """
        model_cache_dir = self.get_model_cache_dir(model_name)
        save_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        
        # save图数据和元数据
        save_dict = {
            'graph': graph_data,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f'Saved graph {graph_name} to {save_path}')
    
    def load_graph(self, model_name: str, graph_name: str) -> Tuple:
        """从缓存load图结构
        
        Args:
            model_name: Model name
            graph_name: 图名称
            
        Returns:
            (graph_data, metadata)
        """
        model_cache_dir = self.get_model_cache_dir(model_name)
        load_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        
        if os.path.exists(load_path):
            load_dict = torch.load(load_path)
            self.logger.info(f'Loaded graph {graph_name} from {load_path}')
            return load_dict.get('graph'), load_dict.get('metadata', {})
        else:
            self.logger.info(f'Graph {graph_name} not found in cache')
            return None, None
    
    def has_cache(self, model_name: str, graph_name: str) -> bool:
        """检查缓存是否存在"""
        model_cache_dir = self.get_model_cache_dir(model_name)
        load_path = os.path.join(model_cache_dir, f'{graph_name}.pt')
        return os.path.exists(load_path)


class DualGNNPreprocessor:
    """DualGNN 预处理：build user-user 图"""
    
    def __init__(self, cache_manager: GraphCacheManager):
        self.cache_manager = cache_manager
        self.logger = getLogger()
    
    def build_user_user_graph(self, interaction_matrix: sp.coo_matrix, 
                             user_features: Optional[torch.Tensor] = None,
                             k: int = 40, 
                             construction: str = 'weighted_sum') -> Dict:
        """build user-user KNN 图
        
        Args:
            interaction_matrix: 用户 - 物品交互矩阵
            user_features: 用户特征（optional）
            k: KNN 的 k 值
            construction: build方式 ('weighted_sum', 'knn', etc.)
            
        Returns:
            user_graph_dict: 用户图字典 {user_id: [neighbor_user_ids]}
        """
        self.logger.info(f'Building user-user graph with k={k}, construction={construction}')
        
        n_users = interaction_matrix.shape[0]
        
        # 基于交互build user-user 相似度
        # 使用余弦相似度
        user_item_matrix = interaction_matrix.tocsr()
        
        # compute user-user 余弦相似度
        user_norms = sp.linalg.norm(user_item_matrix, axis=1)
        user_norms[user_norms == 0] = 1  # 避免除零
        
        # 归一化
        user_item_normalized = user_item_matrix.multiply(1.0 / user_norms.reshape(-1, 1))
        
        # compute相似度矩阵
        sim_matrix = user_item_normalized.dot(user_item_normalized.T)
        
        # 转换为稠密矩阵以便进行 topk
        sim_matrix = sim_matrix.toarray()
        
        # 对每个用户找 k 个最相似的用户
        user_graph_dict = {}
        for u in range(n_users):
            sim_scores = sim_matrix[u]
            sim_scores[u] = -np.inf  # 排除自己
            
            top_k_neighbors = np.argsort(sim_scores)[-k:]
            user_graph_dict[u] = top_k_neighbors.tolist()
        
        self.logger.info(f'Built user-user graph with {len(user_graph_dict)} users')
        
        return user_graph_dict
    
    def save_user_graph(self, user_graph_dict: Dict, k: int, construction: str):
        """save user-user 图到缓存"""
        metadata = {
            'k': k,
            'construction': construction,
            'n_users': len(user_graph_dict)
        }
        
        self.cache_manager.save_graph(
            model_name='DualGNN',
            graph_name='user_graph_dict',
            graph_data=user_graph_dict,
            metadata=metadata
        )
    
    def load_user_graph(self, k: int, construction: str) -> Optional[Dict]:
        """从缓存load user-user 图"""
        # 检查是否有匹配的缓存
        graph_data, metadata = self.cache_manager.load_graph('DualGNN', 'user_graph_dict')
        
        if graph_data is not None:
            # 验证parameter是否匹配
            if metadata.get('k') == k and metadata.get('construction') == construction:
                return graph_data
            else:
                self.logger.info('Cache parameters mismatch, rebuilding...')
        
        return None


class FREEDOMPreprocessor:
    """FREEDOM 预处理：build item-item KNN 图"""
    
    def __init__(self, cache_manager: GraphCacheManager):
        self.cache_manager = cache_manager
        self.logger = getLogger()
    
    def build_item_item_knn_graph(self, item_features: torch.Tensor, 
                                  knn_k: int = 10,
                                  mm_image_weight: float = 0.5,
                                  text_features: Optional[torch.Tensor] = None) -> torch.sparse.FloatTensor:
        """build item-item KNN 图
        
        Args:
            item_features: 物品特征（visual）
            knn_k: KNN 的 k 值
            mm_image_weight: visual 和 text 的权重
            text_features: 文本特征（optional）
            
        Returns:
            mm_adj: 多模态邻接矩阵（稀疏张量）
        """
        self.logger.info(f'Building item-item KNN graph with knn_k={knn_k}')
        
        device = item_features.device
        
        # 归一化特征
        context_norm = item_features.div(torch.norm(item_features, p=2, dim=-1, keepdim=True))
        
        # compute相似度矩阵
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        
        # 获取 k 个最近邻
        _, knn_ind = torch.topk(sim, knn_k, dim=-1)
        
        adj_size = sim.size()
        del sim
        
        # build稀疏邻接矩阵
        indices0 = torch.arange(knn_ind.shape[0]).to(device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        # 归一化拉普拉斯
        mm_adj = self.compute_normalized_laplacian(indices, adj_size)
        
        # 如果有文本特征，融合
        if text_features is not None:
            self.logger.info('Fusing text features...')
            text_norm = text_features.div(torch.norm(text_features, p=2, dim=-1, keepdim=True))
            text_sim = torch.mm(text_norm, text_norm.transpose(1, 0))
            _, text_knn_ind = torch.topk(text_sim, knn_k, dim=-1)
            
            text_indices = torch.stack((
                torch.flatten(torch.arange(knn_k).to(device).unsqueeze(0).expand(-1, knn_k)),
                torch.flatten(text_knn_ind)
            ), 0)
            
            text_adj = self.compute_normalized_laplacian(text_indices, adj_size)
            
            # 加权融合
            mm_adj = mm_image_weight * mm_adj + (1.0 - mm_image_weight) * text_adj
        
        self.logger.info(f'Built item-item KNN graph with shape {mm_adj.shape}')
        
        return mm_adj
    
    def compute_normalized_laplacian(self, indices: torch.Tensor, adj_size: torch.Size) -> torch.sparse.FloatTensor:
        """compute归一化拉普拉斯矩阵"""
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    def save_mm_adj(self, mm_adj: torch.sparse.FloatTensor, knn_k: int, mm_image_weight: float):
        """save多模态邻接矩阵到缓存"""
        metadata = {
            'knn_k': knn_k,
            'mm_image_weight': mm_image_weight,
            'shape': list(mm_adj.shape)
        }
        
        self.cache_manager.save_graph(
            model_name='FREEDOM',
            graph_name=f'mm_adj_freedomdsp_{knn_k}_{int(10*mm_image_weight)}',
            graph_data=mm_adj,
            metadata=metadata
        )
    
    def load_mm_adj(self, knn_k: int, mm_image_weight: float) -> Optional[torch.sparse.FloatTensor]:
        """从缓存load多模态邻接矩阵"""
        graph_name = f'mm_adj_freedomdsp_{knn_k}_{int(10*mm_image_weight)}'
        graph_data, metadata = self.cache_manager.load_graph('FREEDOM', graph_name)
        
        if graph_data is not None:
            # 验证parameter
            if metadata.get('knn_k') == knn_k and metadata.get('mm_image_weight') == mm_image_weight:
                return graph_data
        
        return None

# coding: utf-8
# @email  : enoche.chow@gmail.com
"""
MRS 改进版：自动模态发现
扫描数据目录中的所有 *_feat.npy 文件
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    """所有模型的基类"""
    
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        """打印模型parameter"""
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """通用推荐器基类 - 自动发现模态特征"""
    
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # 基础信息
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # 训练configuration
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # 自动发现并load所有模态特征
        self._load_all_modalities(config)
    
    def _load_all_modalities(self, config):
        """
        自动扫描数据目录中的所有 *_feat.npy 文件
        并load为对应的属性：self.v_feat, self.t_feat, self.audio_feat, 等
        """
        if config.get('end2end', False) or not config.get('is_multimodal_model', True):
            return
        
        dataset_path = os.path.abspath(os.path.join(config['data_path'], config['dataset']))
        
        if not os.path.exists(dataset_path):
            print(f"[Warning] Dataset path not found: {dataset_path}")
            return
        
        # 扫描所有 *_feat.npy 和 *_feat.pt 文件
        feat_patterns = [
            os.path.join(dataset_path, '*_feat.npy'),
            os.path.join(dataset_path, '*_feat.pt')
        ]
        
        feat_files = []
        for pattern in feat_patterns:
            feat_files.extend(glob.glob(pattern))
        
        if not feat_files:
            print(f"[Warning] No feature files found in {dataset_path}")
            return
        
        print(f"\n[Auto-Discovery] Found {len(feat_files)} modality feature files:")
        
        # load每个特征文件
        for feat_file in feat_files:
            # 从文件名提取模态名称
            filename = os.path.basename(feat_file)
            modality_name = filename.replace('_feat.npy', '').replace('_feat.pt', '')
            
            # 转换为属性名（例如 image -> v_feat, text -> t_feat）
            attr_name = self._get_modality_attribute_name(modality_name)
            
            try:
                # load特征
                if feat_file.endswith('.npy'):
                    feat_tensor = torch.from_numpy(np.load(feat_file, allow_pickle=True)).type(torch.FloatTensor).to(self.device)
                elif feat_file.endswith('.pt'):
                    feat_tensor = torch.load(feat_file, map_location=self.device)
                else:
                    continue
                
                # 设置为属性
                setattr(self, attr_name, feat_tensor)
                print(f"  ✓ {modality_name} -> {attr_name} (shape: {feat_tensor.shape})")
                
            except Exception as e:
                print(f"  ✗ Failed to load {filename}: {e}")
        
        print()
    
    def _get_modality_attribute_name(self, modality_name: str) -> str:
        """
        将模态名称转换为属性名
        
        常见映射：
        - visual, image, vision, v, img -> visual_feat
        - textual, text, t, txt -> textual_feat
        - audio, a, sound, acoustic -> audio_feat
        - gpt, llm -> gpt_feat
        - caption, cap -> caption_feat
        - knowledge, k, kg -> knowledge_feat
        - 其他 -> {modality_name}_feat
        """
        modality_lower = modality_name.lower()
        
        # 常见模态的映射 - 使用完整名称
        if modality_lower in ['visual', 'image', 'vision', 'v', 'img']:
            return 'visual_feat'
        elif modality_lower in ['textual', 'text', 't', 'txt']:
            return 'textual_feat'
        elif modality_lower in ['audio', 'a', 'sound', 'acoustic']:
            return 'audio_feat'
        elif modality_lower in ['gpt', 'llm']:
            return 'gpt_feat'
        elif modality_lower in ['caption', 'cap']:
            return 'caption_feat'
        elif modality_lower in ['knowledge', 'k', 'kg']:
            return 'knowledge_feat'
        else:
            # 其他模态使用原名
            return f'{modality_lower}_feat'
    
    def get_available_modalities(self):
        """获取所有可用的模态"""
        modalities = []
        for attr in dir(self):
            if attr.endswith('_feat') and getattr(self, attr) is not None:
                modalities.append(attr.replace('_feat', ''))
        return modalities

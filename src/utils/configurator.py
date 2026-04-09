# coding: utf-8
"""
MRS 版本：移除 Mirror Gradient 和 MDVT configuration
保持简单！
"""

import re
import os
import yaml
import torch
from logging import getLogger


class Config(object):
    """configuration器 - 简化版"""
    
    def __init__(self, model=None, dataset=None, config_dict=None):
        """
        initializeconfiguration器
        
        Args:
            model: Model name
            dataset: Dataset name
            config_dict: configuration字典
        """
        if config_dict is None:
            config_dict = {}
            
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        
        # loadconfiguration文件
        self.final_config_dict = self._load_config_files(config_dict)
        
        # 更新代码中的configuration（优先级最高）
        self.final_config_dict.update(config_dict)
        
        # 设置defaultparameter
        self._set_default_parameters()
        
        # initialize设备
        self._init_device()
    
    def _load_config_files(self, config_dict):
        """loadconfiguration文件"""
        file_config_dict = {}
        file_list = []
        
        # configuration层次：overall -> dataset -> model
        cur_dir = os.getcwd()
        configs_dir = os.path.join(cur_dir, 'configs')
        
        file_list.append(os.path.join(configs_dir, "overall.yaml"))
        file_list.append(os.path.join(configs_dir, "dataset", "{}.yaml".format(config_dict['dataset'])))
        file_list.append(os.path.join(configs_dir, "model", "{}.yaml".format(config_dict['model'])))
        
        hyper_parameters = []
        
        for file in file_list:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    fdata = yaml.load(f.read(), Loader=self._build_yaml_loader())
                    if fdata.get('hyper_parameters'):
                        hyper_parameters.extend(fdata['hyper_parameters'])
                    file_config_dict.update(fdata)
            else:
                # configuration文件不存在时不报错，继续运行
                pass
        
        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict
    
    def _build_yaml_loader(self):
        """build YAML load器"""
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader
    
    def _set_default_parameters(self):
        """设置defaultparameter"""
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        
        # 如果 seed 不在超parameter列表中，添加它
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']
    
    def _init_device(self):
        """initialize设备"""
        use_gpu = self.final_config_dict.get('use_gpu', True)
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict.get('gpu_id', 0))
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value
    
    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None
    
    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict
    
    def get(self, key, default=None):
        """获取configuration值"""
        return self.final_config_dict.get(key, default)
    
    def __str__(self):
        """configuration信息字符串"""
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info
    
    def __repr__(self):
        return self.__str__()

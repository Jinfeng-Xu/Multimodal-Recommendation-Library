# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk

import re
import os
import yaml
import torch
from logging import getLogger


class Config(object):
    """Configuration manager - simplified version"""
    
    def __init__(self, model=None, dataset=None, config_dict=None):
        """
        Initialize configuration manager
        
        Args:
            model: Model name
            dataset: Dataset name
            config_dict: Configuration dictionary
        """
        if config_dict is None:
            config_dict = {}
            
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        
        # Load configuration files
        self.final_config_dict = self._load_config_files(config_dict)
        
        # Update with runtime configuration (highest priority)
        self.final_config_dict.update(config_dict)
        
        # Set default parameters
        self._set_default_parameters()
        
        # Initialize device
        self._init_device()
    
    def _load_config_files(self, config_dict):
        """Load configuration files"""
        file_config_dict = {}
        file_list = []
        
        # Configuration hierarchy: overall -> dataset -> model
        cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isdir(os.path.join(cur_dir, 'configs')):
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
                # Continue running if configuration file does not exist
                pass
        
        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict
    
    def _build_yaml_loader(self):
        """Build YAML loader"""
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
        """Set default parameters"""
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict.get('valid_metric', 'Recall@20')
        self.final_config_dict['valid_metric'] = valid_metric
        valid_metric = valid_metric.split('@')[0].lower()
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        
        # Add seed to hyper_parameters if not already present
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']
    
    def _init_device(self):
        """Initialize device"""
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
        """Get configuration value"""
        return self.final_config_dict.get(key, default)
    
    def __str__(self):
        """Configuration information as string"""
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info
    
    def __repr__(self):
        return self.__str__()
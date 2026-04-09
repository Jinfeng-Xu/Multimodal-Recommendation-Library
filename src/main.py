# coding: utf-8
# @email: jinfeng.xu0605@gmail.com / jinfeng@connect.hku.hk
"""
MRLib: Multimodal Recommendation Library
"""

import os
import argparse
from utils.quick_start import quick_start

os.environ['NUMEXPR_MAX_THREADS'] = '48'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VBPR', help='Model name')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='Dataset name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    
    # Visualization configuration
    parser.add_argument('--no-vis', action='store_true', default=False,
                       help='Disable training visualization')

    config_dict = {
        'gpu_id': 0,
        'enable_visualization': True,  # Enable visualization by default
    }

    args, _ = parser.parse_known_args()
    
    # Disable visualization if user specified --no-vis
    if args.no_vis:
        config_dict['enable_visualization'] = False
    
    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

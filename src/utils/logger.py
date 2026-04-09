# coding: utf-8
# @email: enoche.chow@gmail.com

"""
改进的日志系统：
- 按模型划分子文件夹
- 支持训练过程可视化
- 更美观的output格式
"""

import logging
import os
from utils.utils import get_local_time


def init_logger(config):
    """
    initialize日志系统
    
    功能：
    1. 按模型 - Datasets创建子文件夹：log/{model}/{dataset}/
    2. 日志文件名：{model}-{dataset}-{timestamp}.log
    3. 支持彩色output（终端）
    4. 支持Training visualization开关
    """
    # 按模型 - Datasets组织日志文件夹
    LOGROOT = './log/'
    model_dataset_dir = os.path.join(LOGROOT, config['model'], config['dataset'])
    
    if not os.path.exists(model_dataset_dir):
        os.makedirs(model_dataset_dir)

    # 日志文件名
    logfilename = '{}-{}-{}.log'.format(
        config['model'], 
        config['dataset'], 
        get_local_time().replace(' ', '-').replace(':', '-')
    )

    logfilepath = os.path.join(model_dataset_dir, logfilename)

    # 终端output格式（带颜色）
    sfmt = u"%(asctime)-15s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    
    # 文件output格式（详细）
    filefmt = "%(asctime)-15s %(levelname)-8s %(message)s"
    filedatefmt = "%Y-%m-%d %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    # 设置日志级别
    if config.get('state') is None or config.get('state').lower() == 'info':
        level = logging.INFO
    elif config.get('state').lower() == 'debug':
        level = logging.DEBUG
    elif config.get('state').lower() == 'error':
        level = logging.ERROR
    elif config.get('state').lower() == 'warning':
        level = logging.WARNING
    elif config.get('state').lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    # 文件处理器
    fh = logging.FileHandler(logfilepath, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(fileformatter)

    # 终端处理器
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    # configuration根日志器
    logging.basicConfig(
        level=level,
        handlers=[sh, fh]
    )
    
    # 打印日志路径信息
    logging.info(f"📁 Log directory: {model_dataset_dir}")
    logging.info(f"📄 Log file: {logfilename}")
    logging.info(f"📊 Visualization: {'Enabled' if config.get('enable_visualization', False) else 'Disabled'}")

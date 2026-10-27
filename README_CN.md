<div align="center">
  <a href="https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library"><img width="300px" height="auto" src="images/logo.png"></a>
</div>

# MRLib: 多模态推荐系统开源库

MRLib 是一个面向科研的开源多模态推荐系统（Multimodal Recommendation）综合代码库与基准测试平台。

## 🎉 最新动态  
- [2026.10] 🎯 **[更新]**：我们添加了 HSD（TMM 2026）、PNGRec（SIGIR 2026）、AMUR（CIKM 2026）和 IIMRec（ACM MM 2026）。
- [2026.05] 🎯 **[更新]**：我们添加了MSCA（被WWW 2026接受），感谢[@recomall](https://github.com/recomall)的贡献。
- [2026.04] 🎯 **[更新]**：我们正式发布 MRLib，致力于提供多模态推荐领域的统一基准测试与高质量代码实现。
---

## 🌟 核心特性

✨ **自动模态发现 (Automatic Modality Discovery)**
- **零配置**：自动扫描目录下的 `*_feat.npy` 和 `*_feat.pt` 特征文件
- **灵活集成**：原生支持视觉、文本、音频及其他自定义模态
- **动态加载**：根据数据集实际存在的模态按需加载，无需修改代码

💾 **智能图缓存 (Intelligent Graph Caching)**
- **模型专属缓存**：为每个模型生成独立的缓存目录，避免冲突
- **参数校验**：自动验证缓存参数与当前运行配置是否匹配
- **元数据管理**：完整记录图构建超参数，保障实验可复现性

📊 **实时可视化 (Real-time Visualization)**
- **训练指标监控**：实时绘制 Loss 与各项评估指标曲线
- **最佳模型追踪**：自动记录并高亮最佳 Epoch

🔄 **持续更新 (Continuous Updates)**
- **前沿 SOTA**：定期集成来自顶会/顶刊的最新模型
- **积极维护**：持续修复 Bug、优化训练效率与内存占用
- **社区驱动**：开放贡献通道，欢迎学术界与工业界开发者共同建设

📚 **丰富的数据集支持 (Extensive Dataset Support)**
- **Amazon 系列**：Baby, Sports, Clothing, Pet, Office, Toys, Beauty 等
- **视频/短视频系列**：TikTok, Microlens
- **自定义数据集**：提供清晰的数据格式规范，轻松接入私有数据

---

## 📋 支持的模型列表

按发表年份排序。更多论文列表请参考：[Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)

| 序号 | 模型     | 论文完整标题                                                 | 发表会议/期刊      | 年份 | 链接                                                         |
| :--: | :------- | :----------------------------------------------------------- | :----------------- | :--: | :----------------------------------------------------------- |
| 1 | **VBPR** | VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback | AAAI | 2016 | [论文](https://arxiv.org/pdf/1510.01784) |
| 2 | **MMGCN** | MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video | ACM MM | 2019 | [论文](https://dl.acm.org/doi/10.1145/3343031.3351034) |
| 3 | **GRCN** | Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback | ACM MM | 2020 | [论文](https://dl.acm.org/doi/10.1145/3394171.3413556) |
| 4 | **LATTICE** | Mining Latent Structures for Multimedia Recommendation | ACM MM | 2021 | [论文](https://dl.acm.org/doi/10.1145/3474085.3475259) |
| 5 | **DualGNN** | DualGNN: Dual Graph Neural Network for Multimedia Recommendation | IEEE TMM | 2021 | [论文](https://ieeexplore.ieee.org/document/9662655) |
| 6 | **SLMRec** | Self-Supervised Learning for Multimedia Recommendation | IEEE TMM | 2022 | [论文](https://ieeexplore.ieee.org/document/9811387) |
| 7 | **BM3** | Bootstrap Latent Representations for Multi-modal Recommendation | WWW | 2023 | [论文](https://arxiv.org/pdf/2207.05969) |
| 8 | **MMSSL** | Multi-Modal Self-Supervised Learning for Recommendation | WWW | 2023 | [论文](https://arxiv.org/pdf/2302.10632) |
| 9 | **FREEDOM** | A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation | ACM MM | 2023 | [论文](https://arxiv.org/pdf/2211.06924) |
| 10 | **MGCN** | Multi-View Graph Convolutional Network for Multimedia Recommendation | ACM MM | 2023 | [论文](https://arxiv.org/pdf/2308.03588) |
| 11 | **DRAGON** | Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation | ECAI | 2023 | [论文](https://arxiv.org/pdf/2301.12097) |
| 12 | **LGMRec** | LGMRec: Local and Global Graph Learning for Multimodal Recommendation | AAAI | 2024 | [论文](https://arxiv.org/pdf/2312.16400) |
| 13 | **DiffMM** | DiffMM: Multi-Modal Diffusion Model for Recommendation | ACM MM | 2024 | [论文](https://arxiv.org/pdf/2406.11781) |
| 14 | **DAMRS** | Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback | KDD | 2024 | [论文](https://dl.acm.org/doi/abs/10.1145/3637528.3671703) |
| 15 | **MENTOR** | MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation | AAAI | 2025 | [论文](https://arxiv.org/pdf/2402.19407) |
| 16 | **PGL** | Mind Individual Information! Principal Graph Learning for Multimedia Recommendation | AAAI | 2025 | [论文](https://ojs.aaai.org/index.php/AAAI/article/view/33429) |
| 17 | **SMORE** | Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation | WSDM | 2025 | [论文](https://arxiv.org/pdf/2412.14978) |
| 18 | **COHESION** | COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation | SIGIR | 2025 | [论文](https://arxiv.org/pdf/2504.04452) |
| 19 | **SSR** | Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation | NeurIPS | 2025 | [论文](https://arxiv.org/pdf/2512.01372) |
| 20 | **HPMRec** | Hypercomplex Prompt-aware Multimodal Recommendation | CIKM | 2025 | [论文](https://arxiv.org/pdf/2508.10753) |
| 21 | **LOBSTER** | LOBSTER: Bilateral global semantic enhancement for multimedia recommendation | Information Fusion | 2026 | [论文](https://www.sciencedirect.com/science/article/pii/S1566253525008401) |
| 22 | **MSCA** | Multi-view Semantic Contrastive Alignment for Multimodal Recommendation | WWW | 2026 | [论文](https://dl.acm.org/doi/abs/10.1145/3774904.3792192) |
| 23 | **HSD** | Hierarchy-Aware Multimodal Distillation for Recommendation | IEEE TMM | 2026 | [论文](https://ieeexplore.ieee.org/abstract/document/11344765/) |
| 24 | **PNGRec** | Same Frequency Begets Shared Interests: Popular-Niche Wavelet Graph Learning for Multimodal Recommendation | SIGIR | 2026 | [论文](https://dl.acm.org/doi/abs/10.1145/3805712.3809608) |
| 25 | **AMUR** | Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation | CIKM | 2026 | [论文](https://arxiv.org/abs/2608.27950) |
| 26 | **IIMRec** | One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation | ACM MM | 2026 | [论文](https://arxiv.org/abs/2607.24607) |

> 模型列表按发表年份排序，将持续同步最新研究成果。

---

## 🚀 快速开始

### 安装
```bash
# 克隆仓库
git clone https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library
cd Multimodal-Recommendation-Library

# 安装已验证的 CUDA 12.1 环境
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 基础使用
```bash
# 使用默认配置运行模型
python src/main.py -m HPMRec -d baby

# 指定 GPU 设备
python src/main.py -m COHESION -d sports --gpu_id 1

# 关闭可视化（节省显存/加速）
python src/main.py -m FREEDOM -d clothing --no-vis
```

### 配置说明
模型超参数通过 `src/configs/model/` 目录下的 YAML 文件管理：
```yaml
# HPMRec.yaml 示例
embedding_size: 64
feat_embed_dim: 64
n_mm_layers: 1
n_layers: [3]
knn_k: 10
mm_image_weight: 0.1
reg_weight: [0.001]
hyper_parameters: ["n_layers", "reg_weight"] # 声明需要网格搜索/调参的参数
```

---

## 📁 数据集格式

### 目录结构
```
data/
└── {dataset_name}/
    ├── {dataset_name}.inter # 用户-物品交互记录（Tab 分隔）
    ├── visual_feat.npy    # 视觉特征（可选）
    ├── textual_feat.npy   # 文本特征（可选）
    └── *_feat.npy         # 其他模态特征（可选）
```

### 交互文件格式 (`{dataset_name}.inter`)
```tsv
userID	itemID	x_label
0	123	0
1	456	1
2	789	2
```

`x_label` 表示数据划分：`0` 为训练集，`1` 为验证集，`2` 为测试集。交互文件名和字段名可在 `src/configs/dataset/` 下对应的数据集 YAML 文件中配置。

### 特征文件规范
- **格式**：`.npy` (NumPy) 或 `.pt` (PyTorch Tensor)
- **维度**：`[物品数量, 特征维度]`
- **命名**：`{模态名称}_feat.{npy|pt}`
- **自动识别**：支持 `visual_feat`, `image_feat` 及任意符合 `*_feat` 规则的文件

---

## 🏗️ 项目架构

```
MRLib/
├── src/
│   ├── main.py              # 程序入口
│   ├── models/              # 模型实现
│   │   ├── hpmrec.py
│   │   ├── cohesion.py
│   │   └── ...
│   ├── utils/
│   │   ├── graph_cache.py   # 图缓存管理
│   │   ├── dataset.py       # 数据预处理
│   │   ├── dataloader.py    # 数据加载器
│   │   ├── visualization.py # 训练可视化
│   │   └── quick_start.py   # 快速启动工具
│   ├── configs/
│   │   └── model/           # 模型配置文件
│   └── log/                 # 训练日志与可视化图表
├── data/                    # 数据集存放目录
│   └── cache/               # 自动生成的图缓存
```

---

## 📊 性能基准测试 ([日志](docs/log))

### Baby 数据集

下表报告按验证集选择出的全局最佳配置在测试集上的指标。🥇/🥈/🥉 表示各指标前三个不同数值，并列结果共享同一奖牌。最佳超参数及已下载源日志的映射见 [results.json](docs/results.json)。

| Model | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | NDCG@50 | Precision@10 | Precision@20 | Precision@50 | MAP@10 | MAP@20 | MAP@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [VBPR](docs/log/VBPR/baby) | 0.0371 | 0.0592 | 0.1098 | 0.0199 | 0.0256 | 0.0358 | 0.0042 | 0.0033 | 0.0025 | 0.0141 | 0.0156 | 0.0172 |
| [MMGCN](docs/log/MMGCN/baby) | 0.0398 | 0.0641 | 0.1140 | 0.0210 | 0.0272 | 0.0373 | 0.0044 | 0.0035 | 0.0025 | 0.0148 | 0.0164 | 0.0180 |
| [GRCN](docs/log/GRCN/baby) | 0.0475 | 0.0776 | 0.1368 | 0.0251 | 0.0328 | 0.0448 | 0.0053 | 0.0043 | 0.0031 | 0.0176 | 0.0197 | 0.0215 |
| [LATTICE](docs/log/LATTICE/baby) | 0.0597 | 0.0898 | 0.1495 | 0.0322 | 0.0400 | 0.0521 | 0.0066 | 0.0050 | 0.0034 | 0.0231 | 0.0252 | 0.0271 |
| [DualGNN](docs/log/DualGNN/baby) | 0.0365 | 0.0588 | 0.1073 | 0.0190 | 0.0247 | 0.0345 | 0.0041 | 0.0033 | 0.0024 | 0.0132 | 0.0147 | 0.0162 |
| [SLMRec](docs/log/SLMRec/baby) | 0.0518 | 0.0774 | 0.1256 | 0.0287 | 0.0353 | 0.0451 | 0.0058 | 0.0043 | 0.0028 | 0.0209 | 0.0226 | 0.0242 |
| [BM3](docs/log/BM3/baby) | 0.0531 | 0.0836 | 0.1481 | 0.0284 | 0.0362 | 0.0492 | 0.0059 | 0.0047 | 0.0033 | 0.0201 | 0.0222 | 0.0243 |
| [MMSSL](docs/log/MMSSL/baby) | 0.0569 | 0.0907 | 0.1543 | 0.0309 | 0.0396 | 0.0524 | 0.0063 | 0.0050 | 0.0035 | 0.0222 | 0.0245 | 0.0265 |
| [FREEDOM](docs/log/FREEDOM/baby) | 0.0622 | 0.0975 | 0.1656 | 0.0339 | 0.0430 | 0.0568 | 0.0068 | 0.0054 | 🥉0.0037 | 0.0245 | 0.0270 | 0.0291 |
| [MGCN](docs/log/MGCN/baby) | 0.0630 | 0.0964 | 0.1615 | 0.0347 | 0.0433 | 0.0564 | 0.0069 | 0.0053 | 0.0036 | 0.0252 | 0.0275 | 0.0296 |
| [DRAGON](docs/log/DRAGON/baby) | 0.0644 | 0.0993 | 0.1668 | 0.0350 | 0.0440 | 0.0577 | 0.0071 | 0.0055 | 🥉0.0037 | 0.0253 | 0.0277 | 0.0298 |
| [LGMRec](docs/log/LGMRec/baby) | 0.0660 | 0.1026 | 0.1680 | 0.0358 | 0.0452 | 0.0585 | 0.0072 | 🥉0.0057 | 🥉0.0037 | 0.0258 | 0.0284 | 0.0305 |
| [DiffMM](docs/log/DiffMM/baby) | 0.0578 | 0.0900 | 0.1550 | 0.0314 | 0.0397 | 0.0528 | 0.0064 | 0.0050 | 0.0035 | 0.0227 | 0.0249 | 0.0269 |
| [DAMRS](docs/log/DAMRS/baby) | 0.0591 | 0.0941 | 0.1583 | 0.0318 | 0.0409 | 0.0538 | 0.0066 | 0.0052 | 0.0035 | 0.0227 | 0.0251 | 0.0271 |
| [MENTOR](docs/log/MENTOR/baby) | 0.0669 | 0.1044 | 0.1690 | 0.0363 | 0.0460 | 0.0591 | 🥉0.0074 | 🥈0.0058 | 🥈0.0038 | 0.0262 | 0.0288 | 0.0309 |
| [PGL](docs/log/PGL/baby) | 0.0622 | 0.0969 | 0.1686 | 0.0332 | 0.0421 | 0.0566 | 0.0068 | 0.0053 | 🥉0.0037 | 0.0236 | 0.0260 | 0.0283 |
| [SMORE](docs/log/SMORE/baby) | 0.0679 | 0.1041 | 0.1738 | 🥉0.0369 | 🥉0.0461 | 🥉0.0602 | 🥈0.0075 | 🥉0.0057 | 🥇0.0039 | 🥉0.0265 | 🥉0.0290 | 🥉0.0312 |
| [COHESION](docs/log/COHESION/baby) | 🥈0.0684 | 🥉0.1045 | 🥉0.1744 | 0.0363 | 0.0456 | 0.0597 | 🥈0.0075 | 🥈0.0058 | 🥇0.0039 | 0.0257 | 0.0282 | 0.0304 |
| [SSR](docs/log/SSR/baby) | 0.0665 | 🥈0.1065 | 🥈0.1758 | 0.0357 | 0.0460 | 0.0600 | 0.0073 | 🥇0.0059 | 🥇0.0039 | 0.0255 | 0.0283 | 0.0305 |
| [HPMRec](docs/log/HPMRec/baby) | 0.0645 | 0.1017 | 0.1718 | 0.0348 | 0.0443 | 0.0585 | 0.0071 | 0.0056 | 🥈0.0038 | 0.0249 | 0.0275 | 0.0297 |
| [LOBSTER](docs/log/LOBSTER/baby) | 0.0550 | 0.0867 | 0.1498 | 0.0290 | 0.0372 | 0.0499 | 0.0061 | 0.0048 | 0.0033 | 0.0204 | 0.0226 | 0.0246 |
| [MSCA](docs/log/MSCA/baby) | 🥉0.0682 | 0.1035 | 0.1738 | 🥈0.0375 | 🥈0.0465 | 🥈0.0608 | 🥈0.0075 | 🥉0.0057 | 🥇0.0039 | 🥈0.0273 | 🥈0.0297 | 🥈0.0319 |
| [AMUR](docs/log/AMUR/baby) | 0.0663 | 0.1001 | 0.1688 | 0.0359 | 0.0445 | 0.0584 | 0.0073 | 0.0055 | 🥉0.0037 | 0.0258 | 0.0280 | 0.0302 |
| [HSD](docs/log/HSD/baby) | 0.0575 | 0.0886 | 0.1503 | 0.0301 | 0.0381 | 0.0506 | 0.0064 | 0.0049 | 0.0034 | 0.0210 | 0.0231 | 0.0251 |
| [IIMRec](docs/log/IIMRec/baby) | 🥇0.0701 | 🥇0.1073 | 🥇0.1764 | 🥇0.0382 | 🥇0.0478 | 🥇0.0618 | 🥇0.0077 | 🥇0.0059 | 🥇0.0039 | 🥇0.0276 | 🥇0.0302 | 🥇0.0324 |
| [PNGRec](docs/log/PNGRec/baby) | 0.0678 | 0.1043 | 0.1721 | 0.0366 | 0.0460 | 0.0597 | 🥈0.0075 | 🥉0.0057 | 🥈0.0038 | 0.0263 | 0.0288 | 0.0309 |


🔜 *待补充（持续集成与测试中）*

---

## 🔧 高级用法

### 评估指标配置
默认评估指标为 `Recall@N` 和 `NDCG@N`（N = 10, 20）。  
完整支持的指标与 Top-K 设置包括：

- **指标**：Recall, NDCG, Precision, MAP
- **K值**：@5, @10, @20, @50

### 自定义模型集成
1. 在 `src/models/` 目录下创建新的 `.py` 文件
2. 继承 `GeneralRecommender` 基类
3. 实现以下四个核心方法：
   - `__init__(self, config, dataloader)`
   - `forward(self, interaction)`
   - `calculate_loss(self, interaction)`
   - `full_sort_predict(self, interaction)`
4. 在 `src/configs/model/` 中添加对应的 YAML 配置文件

### 图缓存管理
```python
from utils.graph_cache import GraphCacheManager

# 初始化缓存管理器
cache_manager = GraphCacheManager(data_path, dataset_name)

# 保存图结构
cache_manager.save_graph(
    model_name='MyModel',
    graph_name='item_graph',
    graph_data=graph_tensor,
    metadata={'knn_k': 10}
)

# 加载图结构
graph_data, metadata = cache_manager.load_graph(
    model_name='MyModel',
    graph_name='item_graph'
)
```

### 自定义可视化
```python
from utils.visualization import TrainingVisualizer

visualizer = TrainingVisualizer(
    model_name='HPMRec',
    dataset='baby',
    enable=True
)

# 记录每个 Epoch 的指标
visualizer.log_epoch(epoch, loss, recall, ndcg)

# 训练结束后保存图表
visualizer.save_plots()
```

---

## 🤝 参与贡献

我们热烈欢迎社区开发者与研究者共同完善 MRLib！

- **修复 Bug**：请直接提交 Pull Request，并在 PR 描述中详细说明问题与修复方案。
- **添加新模型**：若您希望将个人模型纳入本库，请确保：
  1. 遵循本项目现有的代码风格与架构
  2. 同步更新模型列表与相关文档
  3. 提供可复现的 YAML 配置

---

## 📝 引用

如果您在学术研究中使用 MRLib，请引用我们的综述论文 [[TMM2026] MRS Survey](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)

```bibtex
@article{xu2026survey,
  title={A survey on multimodal recommender systems: Recent advances and future directions},
  author={Xu, Jinfeng and Chen, Zheyu and Yang, Shuo and Li, Jinze and Wang, Wei and Hu, Xiping and Hoi, Steven and Ngai, Edith},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 许可证

本项目采用 **MIT License** 开源协议。详情请参阅 [LICENSE](LICENSE.txt) 文件。

---

## 🙏 致谢

- 论文列表参考：[Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)
- 本项目代码架构基于开源项目 [MMRec](https://github.com/enoche/MMRec) 改进开发。

---

## 📬 联系方式

- **问题反馈**：请在 GitHub 提交 [Issue](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library/issues)
- **社区讨论**：[GitHub Discussions](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library/discussions)

---
 [🇬🇧 English Version](README.md)

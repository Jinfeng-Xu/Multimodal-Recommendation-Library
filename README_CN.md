# MRS: 面向科研的多模态推荐系统

**MRS** 是一个面向科研的多模态推荐框架，集成 21+ 个 SOTA 模型，提供自动模态发现、智能缓存、实时可视化等先进特性。

---

## 🌟 核心特性

### ✨ 自动模态发现
- **零配置**: 自动扫描 `*_feat.npy` 和 `*_feat.pt` 文件
- **灵活命名**: 支持 visual, image, v, text, t, audio, gpt, caption 等
- **多模态融合**: 无缝处理视觉、文本、音频等多种模态
- **动态加载**: 仅加载每个数据集可用的模态

### 💾 智能图缓存
- **模型专用缓存**: 每个模型有独立的缓存目录
- **参数验证**: 验证缓存参数与当前配置匹配
- **元数据管理**: 存储图构建参数
- **60 倍加速**: 图构建时间从 30 秒降至 0.5 秒

### 📊 实时可视化
- **训练指标**: 实时绘制 loss、Recall、NDCG 曲线
- **最优模型追踪**: 自动识别并可视化最优 epoch
- **专业图表**: 适用于论文的高质量图表
- **零开销**: 可视化异步运行

### 🔄 持续更新
- **最新模型**: 定期集成顶级会议/期刊的 SOTA 模型
- **积极维护**: 持续修复 bug 和优化性能
- **社区驱动**: 欢迎研究人员贡献

### 📚 丰富的数据集支持
- **Amazon 系列**: Baby, Sports, Clothing, Beauty, Tools, Patio
- **Yelp**: 商家评论数据集
- **自定义数据集**: 提供清晰的数据格式规范

---

## 📋 支持的模型列表

*按发表年份排序。参考：[Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)*

| # | 模型 | 论文全称 | 会议/期刊 | 年份 | 链接 |
|---|------|----------|-----------|------|------|
| 1 | **VBPR** | VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback | AAAI | 2016 | [arXiv](https://arxiv.org/pdf/1510.01784) |
| 2 | **MMGCN** | Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video | ACM MM | 2019 | [DOI](https://dl.acm.org/doi/10.1145/3343031.3351034) |
| 3 | **GRCN** | GRCN: Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback | ACM MM | 2020 | [DOI](https://dl.acm.org/doi/10.1145/3394171.3413556) |
| 4 | **LATTICE** | Mining Latent Structures for Multimedia Recommendation | ACM MM | 2021 | [DOI](https://dl.acm.org/doi/10.1145/3474085.3475259) |
| 5 | **DualGNN** | DualGNN: Dual Graph Neural Network for Multimedia Recommendation | IEEE TMM | 2021 | [DOI](https://ieeexplore.ieee.org/document/9662655) |
| 6 | **SLMRec** | Self-Supervised Learning for Multimedia Recommendation | IEEE TMM | 2022 | [DOI](https://ieeexplore.ieee.org/document/9811387) |
| 7 | **BM3** | Bootstrap Multimodal Recommendation | WWW | 2023 | [arXiv](https://arxiv.org/pdf/2207.05969) |
| 8 | **MMSSL** | Multi-Modal Self-Supervised Learning for Recommendation | WWW | 2023 | [arXiv](https://arxiv.org/pdf/2302.10632) |
| 9 | **FREEDOM** | A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation | ACM MM | 2023 | [arXiv](https://arxiv.org/pdf/2211.06924) |
| 10 | **MGCN** | Multi-View Graph Convolutional Network for Multimedia Recommendation | ACM MM | 2023 | [arXiv](https://arxiv.org/pdf/2308.03588) |
| 11 | **DRAGON** | Dragon: A Dual Graph Neural Network for Multimedia Recommendation | ECAI | 2023 | [arXiv](https://arxiv.org/pdf/2301.12097) |
| 12 | **LGMRec** | Light Graph Convolution for Multimedia Recommendation | AAAI | 2024 | [arXiv](https://arxiv.org/pdf/2312.16400) |
| 13 | **DiffMM** | DiffMM: Multi-Modal Diffusion Model for Recommendation | ACM MM | 2024 | [arXiv](https://arxiv.org/pdf/2406.11781) |
| 14 | **DAMRS** | Dual-View Adaptive Multimodal Recommendation System | KDD | 2024 | [DOI](https://dl.acm.org/doi/abs/10.1145/3637528.3671703) |
| 15 | **MENTOR** | MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation | AAAI | 2025 | [arXiv](https://arxiv.org/pdf/2402.19407) |
| 16 | **PGL** | Mind Individual Information! Principal Graph Learning for Multimedia Recommendation | AAAI | 2025 | [DOI](https://ojs.aaai.org/index.php/AAAI/article/view/33429) |
| 17 | **SMORE** | Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation | WSDM | 2025 | [arXiv](https://arxiv.org/pdf/2412.14978) |
| 18 | **COHESION** | Cohesive Hypergraph Learning for Multimedia Recommendation | SIGIR | 2025 | [arXiv](https://arxiv.org/pdf/2504.04452) |
| 19 | **HPMRec** | Hypercomplex Prompt-aware Multimodal Recommendation | CIKM | 2025 | [arXiv](https://arxiv.org/pdf/2508.10753) |
| 20 | **LOBSTER** | LOBSTER: Learning tO BoosTER Multimodal Recommendation | Information Fusion | 2026 | [DOI](https://www.sciencedirect.com/science/article/pii/S1566253525008401) |

*模型按发表年份排序。表格持续更新最新研究成果。*

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/MRS.git
cd MRS

# 安装依赖
pip install -r requirements.txt
```

### 基本用法

```bash
# 默认设置运行
python src/main.py -m HPMRec -d baby

# 指定 GPU
python src/main.py -m COHESION -d sports --gpu_id 1

# 关闭可视化
python src/main.py -m FREEDOM -d clothing --no-vis
```

### 配置说明

模型通过 `src/configs/model/` 目录下的 YAML 文件配置：

```yaml
# HPMRec.yaml
embedding_size: 64
feat_embed_dim: 64
n_mm_layers: 1
n_layers: [3]
knn_k: 10
mm_image_weight: 0.1
reg_weight: [0.001]
hyper_parameters: ["n_layers", "reg_weight"]
```

---

## 📁 数据集格式

### 必需文件

```
data/
└── {dataset_name}/
    ├── inter.csv          # 用户 - 物品交互
    ├── visual_feat.npy    # 视觉特征（可选）
    ├── textual_feat.npy   # 文本特征（可选）
    └── *_feat.npy         # 其他特征（可选）
```

### 交互文件格式

```csv
user_id,item_id,rating,label
0,123,5,1
1,456,4,1
2,789,5,1
```

### 特征文件

- **格式**: `.npy` 或 `.pt` (PyTorch 张量)
- **形状**: `[num_items, feature_dim]`
- **命名**: `{modality}_feat.{npy|pt}`

**自动发现支持**：
- `visual_feat`, `image_feat`
- 任意自定义 `*_feat.npy` 文件

---

## 🏗️ 架构设计

```
MRS/
├── src/
│   ├── main.py              # 入口程序
│   ├── models/              # 模型实现
│   │   ├── hpmrec.py
│   │   ├── cohesion.py
│   │   └── ...
│   ├── utils/
│   │   ├── graph_cache.py   # 图缓存
│   │   ├── dataset.py       # 数据处理
│   │   ├── dataloader.py    # 数据加载
│   │   ├── visualization.py # 训练可视化
│   │   └── quick_start.py   # 快速启动工具
│   ├── configs/
│   │   └── model/           # 模型配置
│   └── log/                 # 训练日志和可视化
├── data/                    # 数据集
│   └── cache/               # 图缓存
```

---

## 📊 性能基准

TODO

---

## 🔧 高级用法

### 自定义模型集成

1. 在 `src/models/` 创建模型文件
2. 继承 `GeneralRecommender` 基类
3. 实现必需方法：
   - `__init__(self, config, dataloader)`
   - `forward(self, interaction)`
   - `calculate_loss(self, interaction)`
   - `full_sort_predict(self, interaction)`
4. 添加配置 YAML 文件

### 图缓存管理

```python
from utils.graph_cache import GraphCacheManager

# 初始化
cache_manager = GraphCacheManager(data_path, dataset_name)

# 保存图
cache_manager.save_graph(
    model_name='MyModel',
    graph_name='item_graph',
    graph_data=graph_tensor,
    metadata={'knn_k': 10}
)

# 加载图
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

# 记录指标
visualizer.log_epoch(epoch, loss, recall, ndcg)

# 保存最终图表
visualizer.save_plots()
```

---

## 🤝 贡献指南

我们欢迎贡献！

### 如何贡献

1. **Fork** 仓库
2. **创建** 特性分支 (`git checkout -b feature/new-model`)
3. **提交** 更改 (`git commit -m 'Add new model'`)
4. **推送** 到分支 (`git push origin feature/new-model`)
5. **创建** Pull Request

### 贡献规范

- 遵循现有代码风格
- 更新文档

---

## 📝 引用

如果您在研究中使用 MRS，请引用我们的综述论文：

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

MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 论文列表参考：[Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)
- 研究社区的贡献
- 用户和贡献者的反馈

---

## 📬 联系方式

- **问题反馈**: 在 GitHub 上提交 issue
- **讨论**: GitHub Discussions

---

**[⬆ 返回顶部](#mrs-面向科研的多模态推荐系统)** | **[🇺🇸 English Version](README.md)**

<div align="center">
  <a href="https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library"><img width="300px" height="auto" src="images/logo.png"></a>
</div>

# MRLib: Multimodal Recommendation Library

 **MRLib** is an open-source, research-oriented library for multimodal recommendation.

> **中文文档:** [🇨🇳 中文版](README_CN.md)

## 🎉 News
- [2026.10]🎯[Update]: We have added HSD (TMM 2026), PNGRec (SIGIR 2026), AMUR (CIKM 2026), and IIMRec (ACM MM 2026).
- [2026.05]🎯[Update]: We have added MSCA (accepted at WWW 2026) to MRLib, thanks to [@recomall](https://github.com/recomall).
- [2026.04]🎯[Update]: We release the MRLib as a comprehensive benchmark and code base for mutlimodal recommendations.
---

## 🌟 Key Features

### ✨ Automatic Modality Discovery
- **Zero Configuration**: Automatically scans `*_feat.npy` and `*_feat.pt` files
- **Flexible Integration**: Supports visual, textual, audio, and other modalities
- **Dynamic Loading**: Loads only available modalities per dataset

### 💾 Intelligent Graph Caching
- **Model-Specific Cache**: Dedicated cache directory per model
- **Parameter Validation**: Verifies cache parameters match configuration
- **Metadata Management**: Stores graph construction parameters

### 📊 Real-time Visualization
- **Training Metrics**: Live plots of loss, and metrics
- **Best Model Tracking**: Auto-identifies best epoch

### 🔄 Continuous Updates
- **Latest SOTA**: Regular integration from top venues
- **Active Maintenance**: Bug fixes and optimizations
- **Community Driven**: Welcoming contributions

### 📚 Extensive Dataset Support
- **Amazon Datasets**: Baby, Sports, Clothing, Pet, Office, Toys, Beauty and etc.
- **Video Datasets**: TikTok and Microlens
- **Custom Datasets**: Clear format specifications

---

## 📋 Supported Models

*Sorted by publication year. Reference: [Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)*

| # | Model | Full Paper Title | Venue | Year | Link |
|---|-------|------------------|-------|------|------|
| 1 | **VBPR** | VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback | AAAI | 2016 | [paper](https://arxiv.org/pdf/1510.01784) |
| 2 | **MMGCN** | MMGCN: Multi-modal Graph Convolution Network for Personalized Recommendation of Micro-video | ACM MM | 2019 | [paper](https://dl.acm.org/doi/10.1145/3343031.3351034) |
| 3 | **GRCN** | Graph-Refined Convolutional Network for Multimedia Recommendation with Implicit Feedback | ACM MM | 2020 | [paper](https://dl.acm.org/doi/10.1145/3394171.3413556) |
| 4 | **LATTICE** | Mining Latent Structures for Multimedia Recommendation | ACM MM | 2021 | [paper](https://dl.acm.org/doi/10.1145/3474085.3475259) |
| 5 | **DualGNN** | DualGNN: Dual Graph Neural Network for Multimedia Recommendation | IEEE TMM | 2021 | [paper](https://ieeexplore.ieee.org/document/9662655) |
| 6 | **SLMRec** | Self-Supervised Learning for Multimedia Recommendation | IEEE TMM | 2022 | [paper](https://ieeexplore.ieee.org/document/9811387) |
| 7 | **BM3** | Bootstrap Latent Representations for Multi-modal Recommendation | WWW | 2023 | [paper](https://arxiv.org/pdf/2207.05969) |
| 8 | **MMSSL** | Multi-Modal Self-Supervised Learning for Recommendation | WWW | 2023 | [paper](https://arxiv.org/pdf/2302.10632) |
| 9 | **FREEDOM** | A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation | ACM MM | 2023 | [paper](https://arxiv.org/pdf/2211.06924) |
| 10 | **MGCN** | Multi-View Graph Convolutional Network for Multimedia Recommendation | ACM MM | 2023 | [paper](https://arxiv.org/pdf/2308.03588) |
| 11 | **DRAGON** | Enhancing Dyadic Relations with Homogeneous Graphs for Multimodal Recommendation | ECAI | 2023 | [paper](https://arxiv.org/pdf/2301.12097) |
| 12 | **LGMRec** | LGMRec: Local and Global Graph Learning for Multimodal Recommendation | AAAI | 2024 | [paper](https://arxiv.org/pdf/2312.16400) |
| 13 | **DiffMM** | DiffMM: Multi-Modal Diffusion Model for Recommendation | ACM MM | 2024 | [paper](https://arxiv.org/pdf/2406.11781) |
| 14 | **DAMRS** | Improving Multi-modal Recommender Systems by Denoising and Aligning Multi-modal Content and User Feedback | KDD | 2024 | [paper](https://dl.acm.org/doi/abs/10.1145/3637528.3671703) |
| 15 | **MENTOR** | MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation | AAAI | 2025 | [paper](https://arxiv.org/pdf/2402.19407) |
| 16 | **PGL** | Mind Individual Information! Principal Graph Learning for Multimedia Recommendation | AAAI | 2025 | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33429) |
| 17 | **SMORE** | Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation | WSDM | 2025 | [paper](https://arxiv.org/pdf/2412.14978) |
| 18 | **COHESION** | COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation | SIGIR | 2025 | [paper](https://arxiv.org/pdf/2504.04452) |
| 19 | **SSR** | Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation | NeurIPS | 2025 | [paper](https://arxiv.org/pdf/2512.01372) |
| 20 | **HPMRec** | Hypercomplex Prompt-aware Multimodal Recommendation | CIKM | 2025 | [paper](https://arxiv.org/pdf/2508.10753) |
| 21 | **LOBSTER** | LOBSTER: Bilateral global semantic enhancement for multimedia recommendation | Information Fusion | 2026 | [paper](https://www.sciencedirect.com/science/article/pii/S1566253525008401) |
| 22 | **MSCA** | Multi-view Semantic Contrastive Alignment for Multimodal Recommendation | WWW | 2026 | [paper](https://dl.acm.org/doi/abs/10.1145/3774904.3792192) |
| 23 | **HSD** | Hierarchy-Aware Multimodal Distillation for Recommendation | IEEE TMM | 2026 | [paper](https://ieeexplore.ieee.org/abstract/document/11344765/) |
| 24 | **PNGRec** | Same Frequency Begets Shared Interests: Popular-Niche Wavelet Graph Learning for Multimodal Recommendation | SIGIR | 2026 | [paper](https://dl.acm.org/doi/abs/10.1145/3805712.3809608) |
| 25 | **AMUR** | Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation | CIKM | 2026 | [paper](https://arxiv.org/abs/2608.27950) |
| 26 | **IIMRec** | One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation | ACM MM | 2026 | [paper](https://arxiv.org/abs/2607.24607) |

*Models sorted by publication year. Table continuously updated with latest research.*

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library
cd Multimodal-Recommendation-Library

# Install the tested CUDA 12.1 stack
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with default settings
python src/main.py -m HPMRec -d baby

# Specify GPU
python src/main.py -m COHESION -d sports --gpu_id 1

# Disable visualization
python src/main.py -m FREEDOM -d clothing --no-vis
```

### Configuration

Models configured via YAML files in `src/configs/model/`:

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

## 📁 Dataset Format

### Required Files

```
data/
└── {dataset_name}/
    ├── {dataset_name}.inter # User-item interactions (tab-separated)
    ├── visual_feat.npy    # Visual features (optional)
    ├── textual_feat.npy   # Text features (optional)
    └── *_feat.npy         # Other features (optional)
```

### Interaction File Format (`{dataset_name}.inter`)

```tsv
userID	itemID	x_label
0	123	0
1	456	1
2	789	2
```

`x_label` identifies the split: `0` for training, `1` for validation, and `2` for testing. The exact interaction filename and field names can be changed in the corresponding dataset YAML file under `src/configs/dataset/`.

### Feature Files

- **Format**: `.npy` or `.pt` (PyTorch tensor)
- **Shape**: `[num_items, feature_dim]`
- **Naming**: `{modality}_feat.{npy|pt}`

**Automatic discovery supports**:
- `visual_feat`, `image_feat`
- Any custom `*_feat.npy` files

---

## 🏗️ Architecture

```
MRS/
├── src/
│   ├── main.py              # Entry point
│   ├── models/              # Model implementations
│   │   ├── hpmrec.py
│   │   ├── cohesion.py
│   │   └── ...
│   ├── utils/
│   │   ├── graph_cache.py   # Graph caching
│   │   ├── dataset.py       # Data processing
│   │   ├── dataloader.py    # Data loading
│   │   ├── visualization.py # Training visualization
│   │   └── quick_start.py   # Quick start utility
│   ├── configs/
│   │   └── model/           # Model configurations
│   └── log/                 # Training logs & visualizations
├── data/                    # Datasets
│   └── cache/               # Graph caches
```

---

## 📊 Performance Benchmarks ([Log](docs/log))

### Baby dataset


The table reports test-set metrics for the validation-selected global-best configuration. 🥇/🥈/🥉 mark the top three distinct values in each metric; ties share the same medal. Best hyperparameters and the downloaded source-log mapping are in [results.json](docs/results.json).

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

🔜 Results for the other datasets are coming soon.

---

## 🔧 Advanced Usage

### Evaluation Configuration

Default evaluation metrics include: Recall@N and NDCG@N with N = 10 or 20.
Whole settings include:

   - Recall, NDCG, Precision, MAP
   - @5, @10, @20, @50

### Custom Model Integration

1. Create model file in `src/models/`
2. Inherit from `GeneralRecommender`
3. Implement required methods:
   - `__init__(self, config, dataloader)`
   - `forward(self, interaction)`
   - `calculate_loss(self, interaction)`
   - `full_sort_predict(self, interaction)`
4. Add configuration YAML

### Custom Model Integration

1. Create model file in `src/models/`
2. Inherit from `GeneralRecommender`
3. Implement required methods:
   - `__init__(self, config, dataloader)`
   - `forward(self, interaction)`
   - `calculate_loss(self, interaction)`
   - `full_sort_predict(self, interaction)`
4. Add configuration YAML

### Graph Cache Management

```python
from utils.graph_cache import GraphCacheManager

# Initialize
cache_manager = GraphCacheManager(data_path, dataset_name)

# Save graph
cache_manager.save_graph(
    model_name='MyModel',
    graph_name='item_graph',
    graph_data=graph_tensor,
    metadata={'knn_k': 10}
)

# Load graph
graph_data, metadata = cache_manager.load_graph(
    model_name='MyModel',
    graph_name='item_graph'
)
```

### Custom Visualization

```python
from utils.visualization import TrainingVisualizer

visualizer = TrainingVisualizer(
    model_name='HPMRec',
    dataset='baby',
    enable=True
)

# Log metrics
visualizer.log_epoch(epoch, loss, recall, ndcg)

# Save final plots
visualizer.save_plots()
```

---



## 🤝 Contributing

We welcome contributions!

### Fix Bug

You can directly propose a pull request and add detailed descriptions to the comment

### Add New Model

If you want to add your model to MRLib, please

- Follow existing code style
- Update documentation

---

## 📝 Citation

If you use MRS in your research, please cite our survey [[TMM2026] MRS Survey](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)

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

## 📄 License

MIT License - see [LICENSE](LICENSE.txt) file for details.

---

## 🙏 Acknowledgments

- Paper list reference: [Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)
- The structure of this code is based on [MMRec](https://github.com/enoche/MMRec)

---

## 📬 Contact

- **Issues**: Open an [Issue](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library/issues) on GitHub
- **Discussion**: [GitHub Discussions](https://github.com/Jinfeng-Xu/Multimodal-Recommendation-Library/discussions)

---

**[🇨🇳 中文版](README_CN.md)**

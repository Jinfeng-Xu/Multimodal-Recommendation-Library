# MRS: Multimodal Recommendation System for Research

**MRS** is a research-oriented multimodal recommendation framework integrating 21+ SOTA models with automatic modality discovery, intelligent caching, and real-time visualization.

---

## 🌟 Key Features

### ✨ Automatic Modality Discovery
- **Zero Configuration**: Automatically scans `*_feat.npy` and `*_feat.pt` files
- **Flexible Naming**: Supports visual, image, audio, etc.
- **Multi-Modal Fusion**: Seamlessly handles visual, textual, audio, and other modalities
- **Dynamic Loading**: Loads only available modalities per dataset

### 💾 Intelligent Graph Caching
- **Model-Specific Cache**: Dedicated cache directory per model
- **Parameter Validation**: Verifies cache parameters match configuration
- **Metadata Management**: Stores graph construction parameters

### 📊 Real-time Visualization
- **Training Metrics**: Live plots of loss, Recall, NDCG
- **Best Model Tracking**: Auto-identifies best epoch
- **Publication-Ready**: Professional-quality charts
- **Zero Overhead**: Asynchronous execution

### 🔄 Continuous Updates
- **Latest SOTA**: Regular integration from top venues
- **Active Maintenance**: Bug fixes and optimizations
- **Community Driven**: Welcoming contributions

### 📚 Extensive Dataset Support
- **Amazon Series**: Baby, Sports, Clothing, Beauty, Tools, Patio
- **Yelp**: Business reviews
- **Custom Datasets**: Clear format specifications

---

## 📋 Supported Models

*Sorted by publication year. Reference: [Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)*

| # | Model | Full Paper Title | Venue | Year | Link |
|---|-------|------------------|-------|------|------|
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

*Models sorted by publication year. Table continuously updated with latest research.*

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/MRS.git
cd MRS

# Install dependencies
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
    ├── inter.csv          # User-item interactions
    ├── visual_feat.npy    # Visual features (optional)
    ├── textual_feat.npy   # Text features (optional)
    └── *_feat.npy         # Other features (optional)
```

### Interaction File Format

```csv
user_id,item_id,rating,label
0,123,5,1
1,456,4,1
2,789,5,1
```

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

## 📊 Performance Benchmarks

TODO

---

## 🔧 Advanced Usage

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

### How to Contribute

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/new-model`)
3. **Commit** changes (`git commit -m 'Add new model'`)
4. **Push** to branch (`git push origin feature/new-model`)
5. **Open** Pull Request

### Contribution Guidelines

- Follow existing code style
- Update documentation

---

## 📝 Citation

If you use MRS in your research, please cite our survey :

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

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Paper list reference: [Awesome-Multimodal-Recommender-Systems](https://github.com/Jinfeng-Xu/Awesome-Multimodal-Recommender-Systems)
- Contributions from research community
- Feedback from users and contributors

---

## 📬 Contact

- **Issues**: Open an issue on GitHub
- **Discussion**: GitHub Discussions

---

**[⬆ Back to Top](#mrs-multimodal-recommendation-system-for-research)** | **[🇨🇳 中文版](README_CN.md)**

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

## 📊 Performance Benchmarks ([Log](docs/log))

### Baby dataset


| Model                                | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | Training Time | Inference Time |
| ------------------------------------ | --------- | --------- | ------- | ------- | ------------- | -------------- |
| [VBPR](docs/log/VBPR/baby)           | 0.0372    | 0.0592    | 0.0199  | 0.0256  | 0.36s/epoch   | 0.63s/epoch    |
| [MMGCN](docs/log/MMGCN/baby)       | 0.0414    | 0.0682    | 0.0215  | 0.0284  | 2.18s/epoch   | 0.62s/epoch    |
| [GRCN](docs/log/GRCN/baby)         | 0.0493    | 0.0790    | 0.0258  | 0.0334  | 1.47s/epoch   | 0.58s/epoch    |
| [LATTICE](docs/log/LATTICE/baby)   | 0.0584    | 0.0893    | 0.0314  | 0.0393  | 0.72s/epoch   | 0.59s/epoch    |
| [DualGNN](docs/log/DualGNN/baby)   | 0.0367    | 0.0592    | 0.0192  | 0.0250  | 4.43s/epoch   | 0.59s/epoch    |
| [SLMRec](docs/log/SLMRec/baby)     | 0.0518    | 0.0774    | 0.0287  | 0.0353  | 2.33s/epoch   | 0.60s/epoch    |
| [BM3](docs/log/BM3/baby)           | 0.0536    | 0.0856    | 0.0289  | 0.0371  | 0.61s/epoch   | 0.58s/epoch    |
| [MMSSL](docs/log/MMSSL/baby)       | 0.0559    | 0.0889    | 0.0306  | 0.0391  | 4.64s/epoch   | 0.61s/epoch    |
| [FREEDOM](docs/log/FREEDOM/baby)   | 0.0622    | 0.0977    | 0.0337  | 0.0427  | 0.77s/epoch   | 0.59s/epoch    |
| [MGCN](docs/log/MGCN/baby)         | 0.0629    | 0.0964    | 0.0346  | 0.0433  | 1.15s/epoch   | 0.60s/epoch    |
| [DRAGON](docs/log/DRAGON/baby)     | 0.0637    | 0.1004    | 0.0351  | 0.0445  | 4.48s/epoch   | 0.67s/epoch    |
| [LGMRec](docs/log/LGMRec/baby)     | 0.0652    | 0.1031    | 0.0353  | 0.0450  | 1.47s/epoch   | 0.60s/epoch    |
| [DiffMM](docs/log/DiffMM/baby)     | 0.0578    | 0.0900    | 0.0314  | 0.0397  | 0.91s/epoch   | 0.60s/epoch    |
| [DAMRS](docs/log/DAMRS/baby)       | 0.0578    | 0.0924    | 0.0316  | 0.0406  | 2.56s/epoch   | 0.58s/epoch    |
| [MENTOR](docs/log/MENTOR/baby)     | 🥈0.0670   | 🥉0.1048   | 🥈0.0362 | 🥈0.0459 | 5.70s/epoch   | 0.59s/epoch    |
| [PGL](docs/log/PGL/baby)           | 0.0610    | 0.0960    | 0.0325  | 0.0415  | 0.88s/epoch   | 0.59s/epoch    |
| [SMORE](docs/log/SMORE/baby)       | 🥇0.0678   | 0.1039    | 🥇0.0368 | 🥇0.0460 | 1.39s/epoch   | 0.67s/epoch    |
| [COHESION](docs/log/COHESION/baby) | 🥈0.0670   | 🥈0.1050   | 0.0350  | 0.0447  | 4.53s/epoch   | 0.59s/epoch    |
| [SSR](docs/log/SSR/baby)           | 🥉0.0665   | 🥇0.1065   | 0.0357  | 🥇0.0460 | 9.80s/epoch   | 0.81s/epoch    |
| [HPMRec](docs/log/HPMRec/baby)     | 0.0660    | 0.1024    | 🥉0.0360 | 🥉0.0453 | 6.15s/epoch   | 2.07s/epoch    |
| [LOBSTER](docs/log/LOBSTER/baby)   | 0.0551    | 0.0861    | 0.0296  | 0.0376  | 1.39s/epoch   | 0.61s/epoch    |

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

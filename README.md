# LP-FT-LoRA: A Three-Stage PEFT Framework for Efficient Domain Adaptation in Bangla NLP Tasks

[![Paper](https://img.shields.io/badge/Paper-ACL%20Submission-blue)](https://arxiv.org/abs/XXXX.XXXXX) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **LP-FT-LoRA**: A novel three-stage Parameter-Efficient Fine-Tuning (PEFT) framework that decouples classifier head alignment from representation learning to enhance stability and performance in domain adaptation tasks.

---


## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training Configuration](#training-configuration)
- [Quick Start](#quick-start)
- [Experimental Results](#experimental-results)
- [Model Architectures](#model-architectures)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [License](#license)


---

## Overview

<figure align="center">
  <img src="./arch.jpg" alt="LP-FT-LoRA Training Framework" width="1000"/>
  <figcaption>Figure: Overview of the LP-FT-LoRA Framework showing frozen backbone, adapter training, and classifier head refinement stages.</figcaption>
</figure>

Standard LoRA fine-tuning jointly trains adapters and classifier heads from random initialization, which can lead to training instability as the classifier "chases" shifting feature representations. **LP-FT-LoRA** addresses this by introducing a structured three-stage approach:

1. **Stage 1 (Linear Probing)**: Train only the classifier head with frozen backbone
2. **Stage 2 (LoRA Probing)**: Train only LoRA adapters with frozen backbone and classifier
3. **Stage 3 (Joint Fine-Tuning)**: Jointly refine both LoRA adapters and classifier head

Our extensive experiments across **five Bangla NLP benchmarks** and **four transformer models** (360M–1.5B parameters) demonstrate that LP-FT-LoRA consistently outperforms standard LoRA fine-tuning and achieves state-of-the-art results.

---

## Key Features

- **Improved Training Stability**: Decouples head alignment from adapter learning to reduce gradient noise
- **Superior Performance**: Achieves 0.17–4.60 percentage points improvement over standard LoRA
- **Parameter Efficient**: Trains only a small fraction of model parameters (~1.2% trainable)
- **Cross-Domain Generalization**: Demonstrates improved robustness on out-of-distribution datasets
- **Flexible Implementation**: Compatible with HuggingFace Transformers and PEFT libraries
- **Multi-Task Support**: Evaluated on fake news detection, sarcasm detection, sentiment analysis, and emotion recognition

---

## Methodology

### Stage 1: Linear Probing

The first stage trains only the classifier head $\phi_C$ on top of the frozen backbone $\phi_{LM}$. This leverages the pretrained representations without modifying them, allowing the classifier to quickly adapt to the downstream task by finding a near-optimal linear decision boundary. By fixing the backbone, the feature space remains stable, mitigating noisy gradients and feature shifts.

### Stage 2: LoRA Linear Probing

In the second stage, the LoRA adapters $\phi_{LoRA}$ are trained while both the backbone $\phi_{LM}$ and the classifier $\phi_C$ remain frozen at their current states. This allows learning task- or domain-specific adjustments to the internal representations via low-rank parameter updates without destabilizing the classifier. The separation reduces convergence issues caused by simultaneously adapting both feature extractors and classifiers.

### Stage 3: Joint Fine-Tuning

The final stage performs joint optimization of both the LoRA adapters $\phi_{LoRA}$ and the classifier $\phi_C$, with the backbone $\phi_{LM}$ frozen. This refinement phase allows the model to synergistically adjust representation learning and classification weights for improved performance, building on the stable initializations from the prior stages. It balances flexibility and stability to achieve better generalization.

---

## Installation

### Requirements
- Python 3.10 or higher
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ GPU memory (tested on Tesla P100)

### Setup

This project is designed to run the entire training and evaluation process inside the provided Jupyter/Colab notebook `experiment.ipynb`. Simply edit the configuration cells inside the notebook to set training parameters, paths, and modes as needed.

### Environment

The notebook installs all dependencies automatically with:

```bash
!pip -q install -U "transformers>=4.44.0" "accelerate>=0.33.0" "datasets>=2.20.0" \
"peft>=0.12.0" "bitsandbytes>=0.43.3" "evaluate" "scikit-learn" "sentencepiece" "huggingface_hub>=0.24.0"
```

Note: Comment out this cell if dependencies are already installed.

## Datasets

We evaluate LP-FT-LoRA on five Bangla NLP benchmarks and two additional cross-dataset evaluation datasets:

### Main Evaluation Datasets

| Dataset | Task | Classes | Size | Domain | Link |
|---------|------|---------|------|--------|------|
| **BanFakeNews** | Fake News Detection | 2 (Authentic/Fake) | ~50K | News Articles | [GitHub](https://github.com/Rowan1697/FakeNews) · [Paper](https://aclanthology.org/2020.lrec-1.349/) |
| **Sarcasm Detection** | Sarcasm Detection | 2 (Sarcastic/Not) | ~50K | News Headlines | [Kaggle](https://www.kaggle.com/competitions/nlp-competition-cuet-ete-day-2022/data) |
| **SentNoB** | Sentiment Analysis | 3 (Pos/Neg/Neu) | 15.7K | Social Media | [GitHub](https://github.com/KhondokerIslam/SentNoB) · [Paper](https://aclanthology.org/2021.findings-emnlp.278/) |
| **Emotion Detection** | Emotion Recognition | 5 Categories | ~15K | YouTube Comments | [Paper](https://ieeexplore.ieee.org/document/8554396) |
| **Sentiment Classification** | Fine-Grained Sentiment | 5 Levels | ~15K | YouTube Comments | [Paper](https://ieeexplore.ieee.org/document/8554396) |

### Cross-Dataset Evaluation

| Dataset | Task | Classes | Size | Link |
|---------|------|---------|------|------|
| **EmoNoBa** | Fine-Grained Emotion | 6 Emotions (Multi-label) | 22.7K | [GitHub](https://github.com/KhondokerIslam/EmoNoBa) · [Paper](https://aclanthology.org/2022.aacl-short.17/) |
| **BanglaSarc** | Sarcasm Detection | 2 (Sarcastic/Not) | 5.1K | [Paper](https://arxiv.org/abs/2209.13461) |

---

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Learning Rate** | 2×10⁻⁴ | Applies to all stages |
| **Optimizer** | AdamW | With cosine LR schedule |
| **Warmup Ratio** | 0.03 | 3% of total training steps |
| **Max Sequence Length** | 512 | Tokens |
| **Batch Size (per device)** | 8 | Effective batch size: 64 (with grad accum) |
| **Gradient Accumulation** | 8 steps | |
| **Training Epochs** | 4 (Stage 1) / 2 (Stages 2-3) | |
| **LoRA Rank (r)** | 16 | Low-rank dimension |
| **LoRA Alpha (α)** | 32 | Scaling factor |
| **LoRA Dropout** | 0.05 | |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Attention + MLP |
| **Attention Implementation** | SDPA | Scaled Dot-Product Attention |

---

Update hyperparameters and paths directly in the notebook configuration block:

```python
# Base model ID
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"

# Training mode
TRAIN_MODE = "LORA_TRAIN_BASE_FREEZE_CLS"  # or other modes defined

# Dataset path
DATA_PATH = "/path/to/your/dataset/"

# Output directory
OUTPUT_DIR = "./output"

# HF Token (optional, for HuggingFace Hub operations)
HF_TOKEN = ""

# Training hyperparameters
NUM_EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 8
GRAD_ACC_STEPS = 8
MAX_SEQ_LEN = 512

# LoRA settings (when using LoRA)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```
---

## Quick Start
### Stage 1: Linear Probing (Train Only Classifier Head)

- Set training mode:

```python
TRAIN_MODE = "FREEZE_BASE_ONLY_CLS"
```

- Since this stage only trains the classifier head on the frozen backbone, no LoRA adapters are used or trained.
- Set output/repo paths accordingly, e.g.:

```python
REPO_ID_IN = ""   # No pretrained heads needed for initialization
REPO_ID_OUT = "your_hf_repo/stage1-linear-probing"
```

- Adjust epochs and learning rate as desired, for example:

```python
NUM_EPOCHS = 4
LR = 2e-4
```

- Dataset and other params remain set as usual.
- After training, heads will be saved and optionally pushed to HF Hub in the output repo.

***

### Stage 2: LoRA Linear Probing (Train Only LoRA Adapters)

- Set training mode:

```python
TRAIN_MODE = "LORA_TRAIN_BASE_FREEZE_CLS"
```

- This stage trains the LoRA adapters only, with the backbone and classifier head frozen.
- You must provide pretrained and saved classifier heads from Stage 1 for initialization:

```python
REPO_ID_IN = "your_hf_repo/stage1-linear-probing"  # Must contain trained classifier heads!
REPO_ID_OUT = "your_hf_repo/stage2-lora-probing"
```

- LoRA hyperparameters (rank, alpha, dropout) must be set as needed:

```python
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

- Epochs typically fewer than Stage 1, e.g.:

```python
NUM_EPOCHS = 2
```

- The model will attach LoRA adapters fresh or load from an existing adapter repo if available.
- After training, save and optionally push adapters and heads.

***

### Stage 3: Joint Fine-Tuning (Train Both LoRA Adapters and Classifier Head)

- Set training mode:

```python
TRAIN_MODE = "LORA_AND_CLS"
```

- This stage fine-tunes both the LoRA adapters and classifier head jointly with the frozen backbone.
- Initialize from output of Stage 2:

```python
REPO_ID_IN = "your_hf_repo/stage2-lora-probing"
REPO_ID_OUT = "your_hf_repo/stage3-joint-finetune"
```

- Typically same LoRA hyperparameters as Stage 2.
- Training epochs similar or fewer, e.g. 2.
- Save and optionally push final adapters and heads to HF Hub.


---


## Experimental Results

### Main Results (Average Accuracy %)

| Model | Params | Linear Probing | Standard LoRA | LoRA Linear Probing | **LP-FT-LoRA** |
|-------|--------|----------------|---------------|---------------------|----------------|
| **SmolLM2-360M** | 360M | 58.39 | 72.06 | 68.04 | **72.64** |
| **Qwen3-0.6B** | 600M | 67.13 | 75.68 | 75.78 | **77.05** |
| **Gemma3-1B** | 1B | 69.71 | 75.93 | 76.83 | **77.00** |
| **Qwen2.5-1.5B** | 1.5B | 63.79 | 75.75 | 75.68 | **75.90** |

### Per-Dataset Results (Gemma3-1B)

| Dataset | Standard LoRA | **LP-FT-LoRA** | Improvement |
|---------|---------------|----------------|-------------|
| **BanFake** | 95.77% | **98.82%** | +3.05% |
| **Sarcasm** | 95.26% | **95.36%** | +0.10% |
| **SentNoB** | 71.69% | **72.07%** | +0.38% |
| **Emotion** | 52.94% | **55.02%** | +2.08% |
| **Sentiment** | 63.97% | **63.75%** | -0.22% |

### Cross-Dataset Generalization (Zero-Shot Transfer)

| Train → Test | Model | Standard LoRA | **LP-FT-LoRA** |
|--------------|-------|---------------|----------------|
| **Sentiment → SentNoB** | Gemma3-1B | 54.22% | **55.55%** |
| **SentNoB → Sentiment** | Gemma3-1B | 56.57% | **60.94%** |
| **Sarcasm → BanglaSarc** | Gemma3-1B | 62.81% | **65.10%** |

---

## Model Architectures

We evaluate LP-FT-LoRA on four open-weight transformer models:

| Model | Parameters | Layers | Hidden Size | Vocab Size | Context Length |
|-------|------------|--------|-------------|------------|----------------|
| **SmolLM2-360M** | 360M | 24 | 960 | 49,152 | 8,192 |
| **Qwen3-0.6B** | 600M | 24 | 1,024 | 151,643 | 32,768 |
| **Gemma3-1B** | 1B | 18 | 2,048 | 256,000 | 8,192 |
| **Qwen2.5-1.5B** | 1.5B | 28 | 1,536 | 151,643 | 32,768 |

### Architecture Overview

```
┌──────────────────────────────────────────┐
│       Frozen Transformer Backbone        │   
│           (φ_LM: 99% params)             │
│  ┌────────────────────────────────────┐  │
│  │  Layer 1: Attention + MLP + LoRA   │  │
│  │  Layer 2: Attention + MLP + LoRA   │  │
│  │            ...                     │  │
│  │  Layer N: Attention + MLP + LoRA   │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
                    ↓
         ┌──────────────────────┐
         │    LoRA Adapters     │
         │ (φ_LoRA: ~1% params) │
         │   Rank r=16, α=32    │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │     Mean Pooling     │
         └──────────────────────┘
                    ↓
         ┌──────────────────────┐
         │ Classification Head  │
         │ (φ_C: <0.1% params)  │
         │    Linear(H → C)     │
         └──────────────────────┘
```

**Training Strategy per Stage:**
- **Stage 1**: Train only the classifier head $\phi_C$ with frozen backbone $\phi_{LM}$ (no LoRA)
- **Stage 2**: Train only the LoRA adapters $\phi_{LoRA}$ with frozen backbone $\phi_{LM}$ and classifier head $\phi_C$
- **Stage 3**: Jointly train both $\phi_{LoRA}$ and $\phi_C$ with frozen backbone $\phi_{LM}$

---

## Repository Structure

```
LP-FT-LoRA/
├── experiment.ipynb                   # Experiment Notebook
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## Citation

If you use this code or our methodology in your research, please cite our paper:

```bibtex
TBA
```

---

## Acknowledgements

### Datasets
- **BanFakeNews**: Hossain et al. (2020) - [Paper](https://aclanthology.org/2020.lrec-1.349/)
- **SentNoB**: Islam et al. (2021) - [Paper](https://aclanthology.org/2021.findings-emnlp.278/)
- **EmoNoBa**: Islam et al. (2022) - [Paper](https://aclanthology.org/2022.aacl-short.17/)
- **BanglaSarc**: Apon et al. (2022) - [Paper](https://arxiv.org/abs/2209.13461)
- **YouTube Emotion/Sentiment**: Irtiza Tripto & Eunus Ali (2018) - [Paper](https://ieeexplore.ieee.org/document/8554396)

### Frameworks
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [PyTorch](https://pytorch.org/)

### Inspiration
This work builds upon:
- **LoRA**: Hu et al. (2022) - [Paper](https://arxiv.org/abs/2106.09685)
- **LP-FT**: Kumar et al. (2022) - [Paper](https://arxiv.org/abs/2202.10054)

---

## Contact

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact the authors at: tomal.tasnimul@gmail.com

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


</div>
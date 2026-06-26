<div align="center">

# 🧬 Autoregressive Boltzmann Generators

[![Paper](http://img.shields.io/badge/Paper-arxiv.2606.27361-B31B1B.svg)](https://arxiv.org/abs/2606.27361)
[![ICML 2026](https://img.shields.io/badge/ICML-2026_Spotlight-blue)](https://icml.cc/virtual/2026)
[![Blog](https://img.shields.io/badge/Blog-ArBG-2ea44f)](https://danyalrehman.com/blogs/arbg/autoregressive_boltzmann_generators.html)
[![Model Weights](https://img.shields.io/badge/Model_Weights-HuggingFace-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/danyalrehman17/robin-transferable)
[![PyTorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<img src="assets/robin.gif" alt="Robin generating equilibrium conformations of unseen peptides" width="480"/>

<sub><em>🧪 Robin sampling equilibrium conformations of unseen peptides, zero-shot.</em></sub>

</div>

> 📄 &nbsp;**Paper Title:** &nbsp;Autoregressive Boltzmann Generators
>
> 👥 &nbsp;**Authors:** &nbsp;Danyal Rehman, Charlie B. Tan, Yoshua Bengio, Avishek Joey Bose, Alexander Tong
>
> 🏛️ &nbsp;**Affiliations:** &nbsp;Mila – Québec AI Institute · Broad Institute of MIT & Harvard · Aithyra · Université de Montréal · University of Oxford · CIFAR · Imperial College London
>
> 📝 &nbsp;**Blog:** &nbsp;[Autoregressive Boltzmann Generators](https://danyalrehman.com/blogs/arbg/autoregressive_boltzmann_generators.html)

This repository is the official implementation of [**Autoregressive Boltzmann Generators**](https://arxiv.org/abs/2606.27361), accepted as a 🏆 **Spotlight** (top 2.2% of submissions) at the **International Conference on Machine Learning (ICML) 2026**.

---

## 🧭 Table of Contents

- [🔭 Overview](#-overview)
- [🧰 Installation](#-installation)
- [📊 Data](#-data)
- [💪 Training](#-training)
- [🔬 Sampling & Evaluation](#-sampling--evaluation)
- [📂 Repository Layout](#-repository-layout)
- [📚 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 License](#-license)

## 🔭 Overview

Efficient sampling of molecular systems at thermodynamic equilibrium is a central challenge in statistical physics. **Boltzmann Generators (BGs)** address it by learning a generative proposal with tractable likelihoods and correcting it against the target energy via importance sampling. Modern BGs rely almost exclusively on **normalizing flows (NFs)**, which either suffer from limited expressivity due to strict invertibility constraints (discrete-time) or expensive likelihoods that require ODE solvers (continuous-time).

**Autoregressive Boltzmann Generators (ArBG)** depart from the flow-based paradigm entirely. ArBG factorizes the molecular density into a sequence of conditionals over Cartesian coordinates,

$$\log p_\theta(x) = \sum_{j=1}^{d} \log p_\theta(x_j \mid x_{\lt j}),$$

modelled by a GPT-style causal transformer operating directly on atoms. This is the first scalable, autoregressive, **diffeomorphism-free** method for Boltzmann Generation, and it brings three advantages:

- ⚡ **Exact, single-pass likelihoods** for self-normalized importance sampling (SNIS) — no Jacobian determinants and no ODE solvers.
- 🌐 **Expressivity without topological constraints** — ArBG can model the disjoint, multi-modal energy landscapes that frustrate flows.
- 🧠 **LLM-style scalability and inference-time control** — temperature scaling for diversity, and **Autoregressive Twisted Sequential Monte Carlo** for substructure-level steering (early rejection of steric clashes before the full molecule is generated).

We discretize each coordinate into uniform bins and predict the bin as a categorical distribution (closely aligned with LLM training), which we find more stable and scalable than continuous mixture parameterizations (MoL/GMM-PixelCNN++). ArBG outperforms flow-based baselines across every single-peptide benchmark, with especially strong scaling to the 10-residue **Chignolin**. We further introduce **Robin**, a 132M-parameter transferable ArBG that generalizes zero-shot to unseen peptides and reduces the energy error (E-W2) on 8-residue systems by **over 60%** relative to the previous state-of-the-art (Prose).

## 🧰 Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and targets Python 3.13 with CUDA 13.x.

```bash
# clone project
git clone https://github.com/danyalrehman/AutoBG.git
cd AutoBG

# create the environment and install dependencies
uv sync
```

### 🔑 Environment variables

Copy the provided template and fill in your values:

```bash
cp .env.example .env
```

`.env` is loaded automatically by `src/train.py` and `src/eval.py`, and is referenced in the Hydra configs via `${oc.env:VAR}`. It defines:

- `SCRATCH_DIR` — directory where datasets, logs, and checkpoints are cached.
- `WANDB_PROJECT` — Weights & Biases project name (defaults to `autoregressive-boltzmann-generators`).
- `WANDB_ENTITY` — your W&B team/username; leave it unset to log to your default account.

## 📊 Data

Both the single-system datasets and ManyPeptidesMD are hosted on Hugging Face.

- [Single systems](https://huggingface.co/datasets/transferable-samplers/sequential-boltzmann-generators-data) — individual peptides used in the paper: alanine dipeptide (`Ace-A-Nme`), tri-alanine (`AAA`), alanine tetrapeptide (`Ace-AAA-Nme`), hexa-alanine (`AAAAAA`), and the decapeptide Chignolin (`GYDPETGTWG`).
- [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md) — the large, diverse peptide dataset used to train the transferable model Robin.

In both cases the codebase automatically downloads the necessary data for training and evaluation. For ManyPeptidesMD the training webdataset is streamed and cached by default.

It is recommended to download data in a single-node setting, as multi-node downloads can fail. On some clusters it is useful to download the data separately first:

```bash
hf download transferable-samplers/many-peptides-md --repo-type dataset --local-dir many-peptides-md
```

## 💪 Training

The codebase builds on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), so experiments are organized into Hydra experiment configuration files under `configs/experiment/`. The autoregressive model is implemented as `AutoregressiveLitModule` (`src/models/autoregressive_module.py`) with a `CausalTransformer` network (`configs/model/net/causal_transformer.yaml`) — a causal-attention transformer with RMSNorm, SwiGLU, FlashAttention, and KV-caching, trained with the Muon optimizer.

We provide three clean launch scripts in [`scripts/`](scripts/) (SLURM `sbatch`). Override any Hydra parameter from the command line.

### 🧪 Single system

Train an ArBG on a single peptide. The example script uses tri-alanine (`AAA`):

```bash
sbatch scripts/train_single_system.sh
```

The same script doubles as an evaluation entry point: set the environment variable `TRAIN=false` to skip training and instead **sample + evaluate** a trained checkpoint (it passes the Hydra flag `train=false` internally, which runs evaluation only, reweighting samples via SNIS to compute E-W2 / T-W2 / TICA-W2):

```bash
TRAIN=false CKPT=/path/to/last.ckpt sbatch scripts/train_single_system.sh
```

To target a different peptide, change the experiment, e.g. `experiment=training/single_system/autoregressive_GYDPETGTWG` for Chignolin.

### 🌐 Transferable (Robin)

Train Robin, the 132M-parameter transferable ArBG, across ManyPeptidesMD (peptides up to 8 residues):

```bash
sbatch scripts/train_transferable.sh
```

## 🔬 Sampling & Evaluation

Evaluation and sampling are run with `src/eval.py`, which loads a checkpoint via `ckpt_path` (a full Lightning `.ckpt`) and reweights samples against the target energy via SNIS to compute E-W2 / T-W2 / TICA-W2.

### 🌐 Transferable (Robin)

The released checkpoint applies **only to the transferable model (Robin)** — the 132M-parameter model hosted on Hugging Face at [`danyalrehman17/robin-transferable`](https://huggingface.co/danyalrehman17/robin-transferable) (`robin.ckpt`). The transferable evaluation script downloads it automatically when no checkpoint is provided, and evaluates all 90 held-out test peptides (30 each of length 2, 4, and 8):

```bash
# evaluate the released Robin checkpoint (auto-downloaded from Hugging Face)
sbatch scripts/eval_transferable.sh

# or evaluate a local checkpoint
CKPT=/path/to/robin.ckpt sbatch scripts/eval_transferable.sh
```

To fetch the checkpoint manually:

```bash
hf download danyalrehman17/robin-transferable robin.ckpt --local-dir checkpoints
```

### 🧪 Single system

There is **no released single-system checkpoint** — the released weights are transferable-only. Train a single-system model (see [Training](#-training)), then evaluate it by setting `TRAIN=false` and pointing `CKPT` at your trained checkpoint:

```bash
TRAIN=false CKPT=/path/to/last.ckpt sbatch scripts/train_single_system.sh
```

## 📂 Repository Layout

```text
.
├── configs/              # Hydra configuration tree
│   ├── experiment/       #   training/ and evaluation/ experiment configs
│   ├── model/            #   model, net, optimizer, and scheduler configs
│   ├── data/             #   single_system/ and transferable/ datamodules
│   └── ...               #   trainer, callbacks, logger, paths, ...
├── src/
│   ├── train.py          # training entry point
│   ├── eval.py           # sampling + evaluation entry point
│   ├── models/           # Lightning modules
│   │   ├── autoregressive_module.py                    # ArBG (single system)
│   │   ├── transferable_boltzmann_generator_module.py  # Robin (transferable)
│   │   ├── neural_networks/                            # CausalTransformer, ...
│   │   └── samplers/                                   # MCMC baselines (HMC/MALA/ULA)
│   ├── data/             # datamodules, datasets, energies, transforms
│   ├── evaluation/       # E-W2 / T-W2 / TICA-W2 metrics and plots
│   ├── optimizers/       # Muon, ...
│   └── utils/
├── scripts/              # SLURM launch scripts (train / eval)
├── assets/               # figures and animations (e.g. robin.gif)
└── pyproject.toml
```

## 📚 Citation

If this codebase is useful towards other research efforts please consider citing us.

```bibtex
@inproceedings{rehman2026arbg,
  title        = {{A}utoregressive {B}oltzmann {G}enerators},
  author       = {Rehman, Danyal and Tan, Charlie B and Bengio, Yoshua and Bose, Joey and Tong, Alexander},
  booktitle    = {International Conference on Machine Learning (ICML)},
  year         = {2026},
  url          = {https://arxiv.org/abs/2606.27361},
  doi          = {10.48550/arXiv.2606.27361},
  eprint       = {2606.27361},
  archivePrefix= {arXiv},
  primaryClass = {cs.LG},
  note         = {Spotlight at ICML 2026}
}
```

## 🙏 Acknowledgements

The authors thank Benjamin Murrell for planting the seeds of this idea, and Luka Mucko for feedback on an early draft. Danyal Rehman received financial support from the Natural Sciences and Engineering Research Council's (NSERC) Banting Postdoctoral Fellowship under Funding Reference No. 198506. The authors acknowledge funding from UNIQUE, CIFAR, NSERC, Intel, and Samsung. The research was enabled in part by computational resources provided by the [Digital Research Alliance of Canada](https://alliancecan.ca), [Mila](https://mila.quebec), [Aithyra](https://www.oeaw.ac.at/aithyra), and [NVIDIA](https://www.nvidia.com). We also thank Hugging Face for hosting the ManyPeptidesMD dataset.

## 📜 License

The core of this repository is licensed under the MIT License (see [LICENSE](./LICENSE)).
Some files include adaptation of third-party code under other licenses (Apple, Meta, NVIDIA, Klein & Noé).
In some cases, these third-party licenses are **non-commercial**.
See [NOTICE](./NOTICE) for details.

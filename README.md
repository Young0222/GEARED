# GEARED

**GEARED** is the official PyTorch implementation of **Efficient Unsupervised Graph Embedding With Attributed Graph Reduction and Dual-Level Loss**.

🔗 **Paper**: [Efficient Unsupervised Graph Embedding With Attributed Graph Reduction and Dual-Level Loss](https://ieeexplore.ieee.org/abstract/document/10616385)

## ✨ Overview

GEARED is an unsupervised graph embedding method designed to improve both **training efficiency** and **embedding quality**.

According to the paper, GEARED combines:

- 🟡 **Attributed graph reduction** to shrink the original graph before contrastive training
- 🔵 **Dual-level loss** to learn better node representations from both node space and feature space
- 🟢 **Adaptive scaling factors** to improve representation separation during training

The paper reports strong results on 14 benchmark datasets, with large speedups on large graphs while keeping strong classification accuracy.

## 🏗️ What This Code Does

The current implementation follows the main pipeline in the paper:

1. Reduce node features with **SVD-based feature reduction**
2. Coarsen the graph with **ASAPooling**
3. Build positive relations with **random-walk-style neighborhood sampling**
4. Train the encoder with **dual-view graph contrastive learning**
5. Evaluate embeddings with a **logistic regression** classifier

## 📦 Environment

This code has been tested with:

- `Python == 3.7.13`
- `PyTorch == 1.12.1`
- `PyTorch Geometric == 2.3.0`

The code also imports these packages during training and evaluation:

- `numpy`
- `scipy`
- `scikit-learn`
- `pyyaml`
- `networkx`
- `ogb`
- `tqdm`

Example installation:

```bash
pip install torch==1.12.1
pip install torch-geometric==2.3.0
pip install numpy scipy scikit-learn pyyaml networkx ogb tqdm
```

## 🗂️ Supported Datasets

The training script currently supports:

- `Cora`
- `CiteSeer`
- `PubMed`
- `DBLP`
- `CS`
- `Physics`
- `Computers`
- `Photo`
- `Wiki`
- `ogbn-arxiv`
- `ogbn-products`

Dataset-specific hyperparameters are defined in [`config.yaml`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/config.yaml).

## 🚀 Quick Start

Run GEARED on the Cora dataset:

```bash
python train_GEARED.py --dataset Cora
```

You can switch to other supported datasets by changing `--dataset`.

## ⚙️ Notes Before Running

- 📁 The script loads datasets from `~/datasets`
- 🧾 Hyperparameters are read from `config.yaml`
- 🖥️ The current training script uses `cuda:4` when CUDA is available

If your machine does not use GPU index `4`, you may need to adjust the device setting in [`train_GEARED.py`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/train_GEARED.py).

## 📈 Example Output

Running:

```bash
python train_GEARED.py --dataset Cora
```

Produces logs like:

```text
(T) | Epoch=001, loss=7.2045, this epoch 0.0183, total 0.0183
(T) | Epoch=002, loss=6.4916, this epoch 0.0127, total 0.0309
(T) | Epoch=003, loss=6.2307, this epoch 0.0109, total 0.0419
...
(T) | Epoch=018, loss=4.4132, this epoch 0.0164, total 0.2542
(T) | Epoch=019, loss=4.3167, this epoch 0.0116, total 0.2658
(T) | Epoch=020, loss=4.7225, this epoch 0.0192, total 0.2850
=== Final ===
ACC mean std:  0.8364727608494921 0.006071757053656014
pre-training time, fune-tuning time: 0.3, 0.3
```

## 🧪 Main Files

- [`train_GEARED.py`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/train_GEARED.py): training and evaluation entry point
- [`model_GEARED.py`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/model_GEARED.py): encoder, projection head, and dual-level loss
- [`eval.py`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/eval.py): downstream evaluation with logistic regression
- [`config.yaml`](/Users/ziyangliu/Desktop/GEARED/GEARED-main/config.yaml): dataset-specific hyperparameters

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{liu2024geared,
  title={Efficient Unsupervised Graph Embedding With Attributed Graph Reduction and Dual-Level Loss},
  author={Liu, Ziyang and Wang, Chaokun and Feng, Hao and Chen, Ziyang},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={36},
  number={12},
  pages={8120--8134},
  year={2024},
  doi={10.1109/TKDE.2024.3436076}
}
```

## 🙌 Acknowledgment

If this project helps your research, a star on GitHub is welcome.

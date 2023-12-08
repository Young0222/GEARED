# GEARED

## This is the PyTorch implementation code for our paper: Efficient Unsupervised Graph Embedding with Attributed Graph Reduction and Dual-level Loss


## Environment Requirements

The code has been tested under Python 3.7.13. The required packages are as follows:

* Pytorch == 1.12.1
* Pytorch Geometric == 2.3.0


## Example : Cora dataset

```python
python train_GEARED.py --dataset Cora
```

The running result is:

*(T) | Epoch=001, loss=8.0443, this epoch 0.0194, total 0.0194
*(T) | Epoch=002, loss=6.8545, this epoch 0.0127, total 0.0322
(T) | Epoch=003, loss=6.3332, this epoch 0.0099, total 0.0421
(T) | Epoch=004, loss=6.0181, this epoch 0.0114, total 0.0535
(T) | Epoch=005, loss=5.7552, this epoch 0.0102, total 0.0637
(T) | Epoch=006, loss=5.7447, this epoch 0.0105, total 0.0742
(T) | Epoch=007, loss=5.4195, this epoch 0.0101, total 0.0842
(T) | Epoch=008, loss=5.1623, this epoch 0.0112, total 0.0955
(T) | Epoch=009, loss=4.5726, this epoch 0.0108, total 0.1062
(T) | Epoch=010, loss=4.5715, this epoch 0.0102, total 0.1164
(T) | Epoch=011, loss=4.0795, this epoch 0.0108, total 0.1272
(T) | Epoch=012, loss=4.1100, this epoch 0.0105, total 0.1376
(T) | Epoch=013, loss=3.3744, this epoch 0.0104, total 0.1481
(T) | Epoch=014, loss=3.2373, this epoch 0.0101, total 0.1582
(T) | Epoch=015, loss=3.1552, this epoch 0.0102, total 0.1684
(T) | Epoch=016, loss=2.7858, this epoch 0.0104, total 0.1788
(T) | Epoch=017, loss=3.1063, this epoch 0.0103, total 0.1891
(T) | Epoch=018, loss=2.7477, this epoch 0.0104, total 0.1994
(T) | Epoch=019, loss=2.6335, this epoch 0.0106, total 0.2101
(T) | Epoch=020, loss=2.8100, this epoch 0.0101, total 0.2202
compression_time: 1.5252
rw_time: 0.0052
pretraining_time: 0.2202
=== Final ===
current time:  0
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.848, F1Ma=0.837, ACC=0.848, time=0.262]
(LR): 100%|█████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.848, F1Ma=0.838, ACC=0.848, time=0.26]
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.849, F1Ma=0.838, ACC=0.849, time=0.253]
(E) | evaluate: micro_f1=0.8481+-0.0004, macro_f1=0.0000+-0.0000, ACC=0.8481+-0.0004, time=0.2582+-0.0037
current time:  1
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.832, F1Ma=0.827, ACC=0.832, time=0.242]
(LR): 100%|█████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.836, F1Ma=0.83, ACC=0.836, time=0.241]
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.832, F1Ma=0.827, ACC=0.832, time=0.244]
(E) | evaluate: micro_f1=0.8333+-0.0016, macro_f1=0.0000+-0.0000, ACC=0.8333+-0.0016, time=0.2423+-0.0013
current time:  2
(LR): 100%|█████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.831, F1Ma=0.819, ACC=0.831, time=0.25]
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.827, F1Ma=0.814, ACC=0.827, time=0.261]
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.831, F1Ma=0.817, ACC=0.831, time=0.291]
(E) | evaluate: micro_f1=0.8296+-0.0020, macro_f1=0.0000+-0.0000, ACC=0.8296+-0.0020, time=0.2674+-0.0171
current time:  3
(LR): 100%|██████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.822, F1Ma=0.81, ACC=0.822, time=0.26]
(LR): 100%|█████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.822, F1Ma=0.81, ACC=0.822, time=0.261]
(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.822, F1Ma=0.809, ACC=0.822, time=0.263]
(E) | evaluate: micro_f1=0.8221+-0.0002, macro_f1=0.0000+-0.0000, ACC=0.8221+-0.0002, time=0.2614+-0.0015
current time:  4
(LR): 100%|███████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.828, ACC=0.84, time=0.26]
(LR): 100%|██████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.829, ACC=0.84, time=0.262]
(LR): 100%|██████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.828, ACC=0.84, time=0.262]
(E) | evaluate: micro_f1=0.8400+-0.0002, macro_f1=0.0000+-0.0000, ACC=0.8400+-0.0002, time=0.2614+-0.0008
mean ACC:  0.8346260387811635
max ACC: 84.8, STD: 0.0
pre-training time, fune-tuning time: 0.2, 0.3

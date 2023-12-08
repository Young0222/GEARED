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

(T) | Epoch=001, loss=8.0443, this epoch 0.0194, total 0.0194

(T) | Epoch=002, loss=6.8545, this epoch 0.0127, total 0.0322

(T) | Epoch=003, loss=6.3332, this epoch 0.0099, total 0.0421

...

(T) | Epoch=018, loss=2.7477, this epoch 0.0104, total 0.1994

(T) | Epoch=019, loss=2.6335, this epoch 0.0106, total 0.2101

(T) | Epoch=020, loss=2.8100, this epoch 0.0101, total 0.2202

pretraining_time: 0.2202

=== Final ===

current time:  0

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.848, F1Ma=0.837, ACC=0.848, time=0.262]

(LR): 100%|█████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.848, F1Ma=0.838, ACC=0.848, time=0.26]

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.849, F1Ma=0.838, ACC=0.849, time=0.253]

(E) | evaluate: micro_f1=0.8481+-0.0004, macro_f1=0.0000+-0.0000, ACC=0.8481+-0.0004, time=0.2582+-0.0037

...

current time:  4

(LR): 100%|███████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.828, ACC=0.84, time=0.26]

(LR): 100%|██████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.829, ACC=0.84, time=0.262]

(LR): 100%|██████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.84, F1Ma=0.828, ACC=0.84, time=0.262]

(E) | evaluate: micro_f1=0.8400+-0.0002, macro_f1=0.0000+-0.0000, ACC=0.8400+-0.0002, time=0.2614+-0.0008

mean ACC:  0.8346260387811635

max ACC: 84.8, STD: 0.0

pre-training time, fune-tuning time: 0.2, 0.3

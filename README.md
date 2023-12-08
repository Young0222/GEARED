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

(T) | Epoch=001, loss=7.2045, this epoch 0.0183, total 0.0183

(T) | Epoch=002, loss=6.4916, this epoch 0.0127, total 0.0309

(T) | Epoch=003, loss=6.2307, this epoch 0.0109, total 0.0419

...

(T) | Epoch=018, loss=4.4132, this epoch 0.0164, total 0.2542

(T) | Epoch=019, loss=4.3167, this epoch 0.0116, total 0.2658

(T) | Epoch=020, loss=4.7225, this epoch 0.0192, total 0.2850

=== Final ===

current time:  0

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.843, F1Ma=0.835, ACC=0.843, time=0.366]

(E) | evaluate: micro_f1=0.8430+-0.0000, macro_f1=0.0000+-0.0000, ACC=0.8430+-0.0000, time=0.3663+-0.0000

current time:  1

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.828, F1Ma=0.817, ACC=0.828, time=0.347]

(E) | evaluate: micro_f1=0.8283+-0.0000, macro_f1=0.0000+-0.0000, ACC=0.8283+-0.0000, time=0.3473+-0.0000

current time:  2

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.834, F1Ma=0.823, ACC=0.834, time=0.312]

(E) | evaluate: micro_f1=0.8343+-0.0000, macro_f1=0.0000+-0.0000, ACC=0.8343+-0.0000, time=0.3125+-0.0000

current time:  3

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.844, F1Ma=0.832, ACC=0.844, time=0.312]

(E) | evaluate: micro_f1=0.8440+-0.0000, macro_f1=0.0000+-0.0000, ACC=0.8440+-0.0000, time=0.3121+-0.0000

current time:  4

(LR): 100%|████████████████████████████████████████████| 300/300 [00:00<00:00, best test F1Mi=0.833, F1Ma=0.825, ACC=0.833, time=0.312]

(E) | evaluate: micro_f1=0.8329+-0.0000, macro_f1=0.0000+-0.0000, ACC=0.8329+-0.0000, time=0.3120+-0.0000

ACC mean std:  0.8364727608494921 0.006071757053656014

pre-training time, fune-tuning time: 0.3, 0.3

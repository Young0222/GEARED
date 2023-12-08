import argparse
import os.path as osp
import random
import pickle as pkl
import networkx as nx
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import sys
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn.functional import conv2d
import torch.nn as nn
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon, WikiCS
from torch_geometric.utils import dropout_adj,dropout_node
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.asap import ASAPooling
from torch_geometric.transforms import SVDFeatureReduction
from ogb.nodeproppred import PygNodePropPredDataset

import copy
import os

from model_GEARED import Encoder, Model, drop_feature
from eval import label_classification, LREvaluator
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.data import GraphSAINTRandomWalkSampler 

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

seed = random.randint(1,999999)
print("pretraining seed: ", seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.cuda.is_available = lambda : False  # using CPU


def train(model: Model, x, edge_index, walks, epoch, dataset, lambda_coe, varepsilon):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    
    loss, tau, quality_node, quality_att = model.loss(lambda_coe, varepsilon, z1, z2, edge_index, walks, epoch)
    loss.backward()
    optimizer.step()

    return loss.item(), tau


def test_LR(encoder_model: Model, x, edge_index, y, seed):
    encoder_model.eval()
    z = encoder_model(x, edge_index)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, y, split, seed)
    return result


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[test_size:],
        'valid': indices[test_size:],
        'test': indices[:test_size]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=4)
    parser.add_argument('--config', type=str, default='config.yaml')


    args = parser.parse_args()

    if torch.cuda.is_available():
        assert args.gpu_id in range(0, 8)
        torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    for _ in range(1, 2):
        learning_rate = config['learning_rate']
        num_hidden = config['num_hidden']
        num_proj_hidden = config['num_proj_hidden']
        activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU(), })[config['activation']]
        base_model = ({'GCNConv': GCNConv})[config['base_model']]
        num_layers = config['num_layers']
        drop_edge_rate_1 = config['drop_edge_rate_1']
        drop_edge_rate_2 = config['drop_edge_rate_2']
        drop_feature_rate_1 = config['drop_feature_rate_1']
        drop_feature_rate_2 = config['drop_feature_rate_2']
        num_epochs = config['num_epochs']
        weight_decay = config['weight_decay']
        tau1 = config['tau1']
        tau2 = config['tau2']
        coarsen_ratio = config['coarsen_ratio']
        reduce_ratio = config['reduce_ratio']
        varepsilon = config['varepsilon']
        lambda_coe = config['lambda_coe']


        def get_dataset(path, name):
            assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'CS', 'Physics', 'Computers', 'Photo', 'Wiki', 'ogbn-arxiv', 'ogbn-products']
            name = 'dblp' if name == 'DBLP' else name
            if name in ['Cora', 'CiteSeer', 'PubMed']: 
                return Planetoid(
                path,
                name)
            elif name in ['CS', 'Physics']:
                return Coauthor(
                path,
                name,
                transform=T.NormalizeFeatures())
            elif name in ['Computers', 'Photo']:
                return Amazon(
                path,
                name,
                transform=T.NormalizeFeatures())
            elif name in ['Wiki']:
                return WikiCS(
                path,
                transform=T.NormalizeFeatures())
            elif name in ["ogbn-arxiv"]:
                return PygNodePropPredDataset(
                    root=path,
                    name="ogbn-arxiv",
                )
            elif name in ["ogbn-products"]:
                return PygNodePropPredDataset(
                    root=path,
                    name="ogbn-products",
                )
            else:
                return CitationFull(
                path,
                name)

        path = osp.join(osp.expanduser('~'), 'datasets')
        print("path: ", path)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
    
        reduce_dimension = int(dataset.num_features*reduce_ratio)

        encoder = Encoder(reduce_dimension, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau1, tau2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # reduce feature dimension
        reduced_data = copy.deepcopy(data)


        start = t()
        if args.dataset not in {"ogbn-arxiv", "ogbn-products"}:
            reduce_feature_method = SVDFeatureReduction(reduce_dimension)
            reduced_data = reduce_feature_method(reduced_data)

        # coarsen the graph
        coarsen_graph_method = ASAPooling(reduced_data.x.shape[1], coarsen_ratio).to(device)
        sample_x, sample_edge_index, _, _, _ = coarsen_graph_method(reduced_data.x, reduced_data.edge_index)
        sample_edge_index = sample_edge_index.to(device)
        sample_x = sample_x.detach().to(device) # the original sample_x is trainable!

        now = t()
        compression_time = now - start
        start = t()

        # random walk to generate positives
        nodes_idx = torch.arange(sample_x.shape[0])
        walk_loader = NeighborSampler(
            sample_edge_index,
            node_idx=nodes_idx,
            sizes=[15,10,5],
            batch_size=102400,
            shuffle=True,
        )
        for batch_size, n_id, adj in walk_loader:
            walks = torch.cat( (adj[0].edge_index, adj[1].edge_index, adj[2].edge_index), 1)
        print("walks: ", walks, walks.shape)

        now = t()
        rw_time = now - start
        start = t()

        print("raw shape: ", data.x.shape, data.edge_index.shape)
        print("coarsen shape: ", sample_x.shape, sample_edge_index.shape)
        
        prev = start
        tau_list = []

        for epoch in range(1, num_epochs + 1):
            loss, tau = train(model, sample_x, sample_edge_index, walks, epoch, args.dataset, lambda_coe, varepsilon)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            tau_list.append(tau)
            prev = now


        pretraining_time = prev - start

        acc_list = []
        print("=== Final ===")
        res_list = []
        std_list = []
        time_list = []

        for i in range(5):
            print("current time: ", i)
            res = test_LR(model, reduced_data.x, reduced_data.edge_index, reduced_data.y, seed)
            res_list.append(res['ACC']['mean'])
            std_list.append(res['ACC']['std'])
            time_list.append(res['time']['mean'])

        print("ACC mean std: ", np.mean(res_list), np.std(res_list))
        print(f'pre-training time, fune-tuning time: {pretraining_time:.1f}, {np.mean(time_list):.1f}')


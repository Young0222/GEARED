import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau1: float = 0.8, tau2: float = 0.2, tau1_att: float = 0.8, tau2_att: float = 0.2):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau1: float = tau1
        self.tau2: float = tau2
        self.tau1_att: float = tau1_att
        self.tau2_att: float = tau2_att
        self.pre_grad: float = 0.0

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def uniform_loss(self, z: torch.Tensor, t: int = 2):
        return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

    def indicator(self, z: torch.Tensor, t: int = 2):
        num_sample = 5
        num = z.shape[0]
        p = torch.ones(num)
        index = p.multinomial(num_samples=num_sample, replacement=True)
        z_sample = z[index]
        total_separation = -torch.pdist(z_sample, p=2).pow(2).mul(-t).exp().mean().log()
        return total_separation

    def momentum_node(self, x_start: float, z: torch.Tensor, step: float = 1e-2, discount: float = 1.0): # step setting---Cora,Citeseer:1e-2; CS,Computers,Photo,Physics:1e-2; Pubmed:1e-3; 
        if x_start <= self.tau2:
            return x_start, 0
        x = x_start
        grad = self.indicator(z).item()

        min_number = 1e-1
        if grad < min_number:
            grad = min_number
        self.pre_grad = self.pre_grad * discount + 1 / grad
        x -= self.pre_grad * step

        return x, grad

    def momentum_att(self, x_start: float, z: torch.Tensor, step: float = 1e-2, discount: float = 1.0): # step setting---Cora,Citeseer:1e-2; CS,Computers,Photo,Physics:1e-2; Pubmed:1e-3; 
        if x_start <= self.tau2:
            return x_start, 0
        x = x_start
        grad = self.indicator(z).item()

        min_number = 1e-1
        if grad < min_number:
            grad = min_number
        self.pre_grad = self.pre_grad * discount + 1 / grad
        # self.pre_grad = 1 / grad
        x -= self.pre_grad * step

        return x, grad


    def torch_cov(self, input_vec: torch.Tensor):    
        x = input_vec- torch.mean(input_vec,axis=0)
        cov_matrix = torch.matmul(x.T, x) / (x.shape[0]-1)
        return torch.det(cov_matrix)

    def semi_loss(self, varepsilon:float, z1: torch.Tensor, z2: torch.Tensor, epoch: int, edge_index: torch.Tensor, walks: torch.Tensor):
        if self.tau1 < 0.0:
            self.tau1 = self.tau2

        if self.tau1_att < 0.0:
            self.tau1_att = self.tau2
        
        f1 = lambda x: torch.exp(x / self.tau1)
        f2 = lambda x: torch.exp(x / self.tau1_att)

        self.tau1, quality_node = self.momentum_node(self.tau1, z1, varepsilon)
        between_sim = f1(self.sim(z1, z2))
        adj = torch.Tensor(to_scipy_sparse_matrix(walks, num_nodes=z1.shape[0]).todense()).to(z1.device)
        positives = between_sim.diag() + torch.mul(between_sim, adj).sum(1)
        total = between_sim.sum(1)

        az1 = z1.t()
        az2 = z2.t()
        self.tau1_att, quality_att = self.momentum_att(self.tau1_att, az1)
        att_between_sim = f2(self.sim(az1, az2))
        att_positives = att_between_sim.diag()
        att_total = att_between_sim.sum(1)

        return - torch.log(positives / total), - torch.log(att_positives / att_total), quality_node, quality_att

    def loss(self, lambda_coe: float, varepsilon: float, z1: torch.Tensor, z2: torch.Tensor, edge_index: torch.Tensor, walks: torch.Tensor, epoch: int =0, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1, l1_att, quality_node, quality_att = self.semi_loss(varepsilon, h1, h2, epoch, edge_index, walks)
        l2, l2_att, _, _ = self.semi_loss(varepsilon, h2, h1, epoch, edge_index, walks)

        ret = (l1 + l2) * 0.5
        ret_att = (l1_att + l2_att) * 0.5
        
        ret = ret.mean() if mean else ret.sum()
        ret_att = ret_att.mean() if mean else ret_att.sum()

        return ret + lambda_coe * ret_att, self.tau1, quality_node, quality_att

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
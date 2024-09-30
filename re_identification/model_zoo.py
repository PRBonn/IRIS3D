import torch
import torch_geometric
from torch_geometric.nn import GCNConv, EdgeConv, global_mean_pool
import numpy as np
import random

from backbone import *

class MinkEncoder(nn.Module):
    def __init__(self, cfg, verbose=False):
        super().__init__()
        self.minkunet = MinkEncoderDecoder(cfg, 1500)
        self.dropout_strawberries = 0
        if "dropout" in cfg:
            self.dropout_strawberries = float(cfg["dropout"])

        if verbose:
            print(f"|***** MinkEncoder *****|")
            print(f"| - channels {cfg['CHANNELS']}")
            print(f"| - strawberries dropout {self.dropout_strawberries}")
            print(f"|***********************|")
            print()

        self.str = str(cfg['CHANNELS']).replace("[", "").replace("]", "").replace(" ", "")

        self.str = f"me{self.str}"

    
    def forward(self, dataloader, dataset):
        dataset.keep_ratio = random.random() * self.dropout_strawberries + (1-self.dropout_strawberries)
        idxs_b, pcds, keys_b = next(iter(dataloader))
        descriptors = self.minkunet(pcds)
        return descriptors, idxs_b, keys_b

class GCN(nn.Module):
    def __init__(self, features_dim, kernel):
        super().__init__()
        self.kernel = kernel
        if kernel=="GCNConv":
            self.conv1 = GCNConv(features_dim, features_dim) #, add_self_loops=True)
            self.conv2 = GCNConv(features_dim, features_dim) #, add_self_loops=True)
        elif kernel=="EdgeConv":
            self.mlp1 = nn.Linear(features_dim*2, features_dim)
            self.mlp2 = nn.Linear(features_dim*2, features_dim)
            self.conv1 = EdgeConv(self.mlp1, aggr="mean")
            self.conv2 = EdgeConv(self.mlp2, aggr="mean")
        else:
            quit("kernel", kernel, "not implemented")

    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        batch      = data.batch

        x = self.conv1(x=x, edge_index=edge_index)
        x = self.conv2(x=x, edge_index=edge_index)
        x = global_mean_pool(x, batch)

        return x

class GCNNeck(nn.Module):
    def __init__(self, desc_len, gcn_cfg, verbose=False):
        super().__init__()
        self.graph_radius = float(gcn_cfg["graph_radius"])
        self.convlayer = gcn_cfg["CONV_LAYER"]
        self.desc_len = desc_len
        self.gcn = GCN(self.desc_len, self.convlayer)

        if verbose:
            print(f"|******* GCNNeck *******|")
            print(f"| - desc_len      {self.desc_len}")
            print(f"| - graph kernel  {self.convlayer}")
            print(f"| - graph_radius  {self.graph_radius}")
            print(f"|***********************|")
            print()

        self.str = f"gcn_{self.convlayer},gr{self.graph_radius}"


    def forward(self, dataset, descriptors, idxs_b, keys_b):
        centers = dataset.centers[idxs_b]
        centered = np.zeros((centers.shape[0], centers.shape[0], 3))
        for r in range(centers.shape[0]):
            centered[r] = centers[r] - centers
        centered = torch.Tensor(centered)

        datas = []
        for anchor_idx, center in enumerate(centers):
            mask = np.linalg.norm(centers - center, axis=1, ord=2) <= self.graph_radius
            mask[anchor_idx] = False
            edges = [[anchor_idx, k] for k, isnn in enumerate(mask) if isnn]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            pos = torch.tensor(centered[anchor_idx]).float().t().contiguous().T
            data = torch_geometric.data(x=descriptors.clone(), edge_index=edge_index, pos=pos).cuda()
            datas.append(data)
        
        loader = torch_geometric.loader.DataLoader(datas, batch_size=len(datas), shuffle=False)
        datasb = next(iter(loader))
        
        nn_descriptors = self.gcn(datasb)
        descriptors = torch.cat((descriptors, nn_descriptors), dim=1)

        return descriptors

class Encoder(nn.Module):
    def __init__(self, cfg, verbose=False):
        super().__init__()
        self.backbone_name = cfg.BACKBONE["NAME"]
        self.neck_name     = cfg.NECK["NAME"]

        self.backbone = None
        self.neck     = None

        ### BACKBONE
        if self.backbone_name=="minkunet":
            self.backbone        = MinkEncoder(cfg.BACKBONE[self.backbone_name])
            self.backbone_outlen = int(cfg.BACKBONE[self.backbone_name]["CHANNELS"][4])
        else:
            quit(f"backbone {self.backbone_name} not implemented!")

        self.str = self.backbone.str


        ### NECK
        if self.neck_name=="None":
            self.descriptor_len = self.backbone_outlen*2
        elif self.neck_name=="gcn":
            self.neck           = GCNNeck(self.backbone_outlen, cfg.NECK_gcn)
            self.descriptor_len = self.backbone_outlen*4                 ## concatenate also gcn
        elif self.neck_name=="midattn":
            self.neck           = MidAttn(self.backbone_outlen*2, cfg.NECK_midattn)
            self.descriptor_len = self.backbone_outlen*6                 ## concatenate also midattn
        else:
            quit(f"neck {self.neck_name} not implemented!")

        if self.neck is not None:
            self.str += "," + self.neck.str

        if verbose:
            print(f"|** Encoder structure **|")
            print(f"| - backbone_name   {self.backbone_name:10s} | backbone_outlen {self.backbone_outlen}")
            print(f"| - neck_name       {self.neck_name:10s} | descriptor_len  {self.descriptor_len}")
            print(f"|                       |")



    def forward(self, dataloader, dataset):
        descriptors, idxs_b, keys_b = self.backbone(dataloader, dataset)
        #print("descriptors", descriptors.shape)

        if self.neck is not None:
            descriptors = self.neck(dataset, descriptors, idxs_b, keys_b)


        return descriptors, idxs_b, keys_b



class SelfAttention(nn.Module):
    def __init__(self, feature_dim, intern_dim, dim_feedforward, nheads):
        super(SelfAttention, self).__init__()
        self.intern_dim = intern_dim
        self.dim_feedforward = dim_feedforward
        self.nheads = nheads

        #print("Matcher. linear", feature_dim, "to", self.intern_dim)
        #print(f"Matcher. TEL   d_model={self.intern_dim} nhead={self.nheads} dim_feedforward={self.dim_feedforward}")

        self.pre = nn.Linear(feature_dim, self.intern_dim, bias=True)
        self.te = nn.TransformerEncoderLayer(d_model=self.intern_dim, nhead=self.nheads, dim_feedforward=self.dim_feedforward, batch_first=False, dropout=0)#0.45)
        self.lin = nn.Linear(self.intern_dim, 1, bias=True)
        self.norm = nn.BatchNorm1d(self.intern_dim)
        self.norm2 = nn.BatchNorm1d(self.intern_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.pre(x)
        x = self.act(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.te(x)
        x = self.act(x)
        x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.lin(x)
        return x.squeeze()


class Matcher(nn.Module):
    def __init__(self, matcher_cfg, descriptor_length, std_err=0.001, device=torch.device("cpu"), verbose=False):
        super().__init__()

        intern_dim      = matcher_cfg["INTERN_DIM"]
        dim_feedforward = matcher_cfg["DIM_FEEDFORWARD"]
        nheads          = matcher_cfg["NUM_HEADS"]
        
        self.selfattention = SelfAttention(descriptor_length, intern_dim, dim_feedforward, nheads)
        self.posenc        = PositionalEncoder(descriptor_length)
        self.descriptor_length = descriptor_length
        self.std_err     = std_err

        if verbose:
            print(f"|** Matcher structure **|")
            print(f"| - descriptor_len     {self.descriptor_length}")
            print(f"| - std_err            {self.std_err}")
            print(f"| - sa.intern_dim      {self.selfattention.intern_dim}")
            print(f"| - sa.dim_feedforward {self.selfattention.dim_feedforward}")
            print(f"| - sa.nheads          {self.selfattention.nheads}")
            print(f"|***********************|")
            print()
        self.str = f"mat{self.selfattention.intern_dim}x{self.selfattention.dim_feedforward}x{self.selfattention.nheads}"

    def getreadyforattention(self, descriptors_from, descriptors_next, next_centers_minus_from_centers, augment):
        """
        ## descriptors_from (gdf): shape [N, C]
        ## descriptors_next (gdn): shape [M, C]
        ##
        ## let's build a two new tensors of shape [M, N, C]
        ##     +-----+
        ##   C/     /|      - the first is the repetition of "next" descs N times, one for each "from" descs
        ##   +-----+ |
        ## M |     | +      - the second is the repetition of "from" descs M times, one  for each "next" descs
        ##   |     |/
        ##   +-----+
        ##      N 
        """
        N, C = descriptors_from.shape
        M, _ = descriptors_next.shape

        ## [M, C] -> [C, M] -> [1, C, M] -> [N, C, M] -> [M, N, C]
        next_rep = descriptors_next.T[None].expand(N, C, M).permute(2, 0, 1)

        ## [N, C] -> [1, N, C] -> [M, N, C]
        from_rep = descriptors_from[None].expand(M, N, C)

        vec = next_centers_minus_from_centers.cuda()
        if augment:
            vec += torch.normal(0, self.std_err, vec.shape).cuda()
        vecenv = self.posenc(vec)

        """
        ##      +-----+
        ##    C/____ /|    1) from descriptors (from_rep)
        ##   C/     /|+    0) next descriptors (next_rep)
        ##   +-----+ |/
        ## M |     | /      
        ##   |     |/      ~~~~ will be altered with position encoding (vecenv)
        ##   +-----+
        ##      N 
        """
        src = torch.cat((next_rep, from_rep), dim=2) + vecenv
        unk = torch.zeros((src.shape[0], 1, src.shape[2])).cuda()

        """
        ##      +0-----+
        ##    C/0____ /|    1) from descriptors (from_rep)
        ##   C/0     /|+    0) next descriptors (next_rep)
        ##   +0-----+ |/
        ## M |0     | /      
        ##   |0     |/      first "column" is for "no match" prediction
        ##   +0-----+
        ##      N 
        """
        return torch.cat((unk, src), dim=1)

    def forward(self, descriptors_from, graph_descriptors_next, next_centered, pos_augment):
        src = self.getreadyforattention(descriptors_from, graph_descriptors_next, next_centered, pos_augment)
        predicted_matrix_logits = self.selfattention(src)
        return predicted_matrix_logits
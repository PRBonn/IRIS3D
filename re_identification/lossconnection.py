import open3d as o3d
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import glob, os


class LossConnection(Dataset):
    def __init__(self, connections_path, dataset_from, dataset_next):

        self.dataset_from = dataset_from
        self.dataset_next = dataset_next

        self.labels_from_connection = torch.zeros(dataset_from.centers.shape[0], dtype=int, device=torch.device("cuda")) -1
        self.labels_next_connection = torch.zeros(dataset_next.centers.shape[0], dtype=int, device=torch.device("cuda")) -1


        # initialization. no fruit is matched with any fruit
        self.connmatrix = torch.zeros((dataset_from.centers.shape[0]+1, dataset_next.centers.shape[0]+1), dtype=torch.float32, device=torch.device("cuda"))
        self.connmatrix[0, :] = 1
        self.connmatrix[:, 0] = 1

        self.connections_keys = {}
        self.connections_idxs = []

        self.conn_from2next = {}

        self.next_centered = np.zeros((dataset_next.centers.shape[0], dataset_from.centers.shape[0], 3))
        for r in range(dataset_next.centers.shape[0]):
            self.next_centered[r] = dataset_next.centers[r] - dataset_from.centers
        self.next_centered = torch.Tensor(self.next_centered)

        self.nomatch_from_num = 0
        self.nomatch_next_num = 0
        self.matched_num = 0

        with open(connections_path) as f:
            connections_json = json.load(f)

            self.conncont = 0

            for k in connections_json:
                key_from = k['first']
                key_next = k['second']

                if key_from not in dataset_from.keys or key_next not in dataset_next.keys:
                    continue

                self.matched_num += 1

                index_of_key_from = dataset_from.keys.index(key_from)
                index_of_key_next = dataset_next.keys.index(key_next)

                self.labels_from_connection[index_of_key_from] = self.conncont
                self.labels_next_connection[index_of_key_next] = self.conncont
                self.conncont += 1

                # we know that this is a match. Thus
                # 1) set to 1 the corresponding element
                # 2) set to 0 the "no patch" pseudo labels (from and to)
                self.connmatrix[index_of_key_from+1, index_of_key_next+1] = 1
                self.connmatrix[0, index_of_key_next+1] = 0
                self.connmatrix[index_of_key_from+1, 0] = 0

                self.connections_keys[key_next] = key_from
                #print(f"{len(self.connections_idxs):3d} connection: from {key_from:4d} to {key_next:4d} {self.connections_idxs.count(key_next)}")
                self.connections_idxs.append(key_next)


                self.conn_from2next[key_from] = key_next

        allmatchedfromkeys = [self.connections_keys[key_next] for key_next in self.connections_keys]

        for i, key in enumerate(dataset_from.keys):
            if key not in allmatchedfromkeys:
                self.nomatch_from_num += 1

        for i, key in enumerate(dataset_next.keys):
            if key not in self.connections_keys:
                self.nomatch_next_num += 1


    def __len__(self):
        return len(self.connections_idxs)

    def __getitem__(self, idx):
        key_next = self.connections_idxs[idx]
        key_from = self.connections_keys[key_next]
        return key_from, key_next

    def getdata(self, from_idxs, next_idxs):

        next_centered = self.next_centered.clone()[next_idxs]
        next_centered = next_centered[:, from_idxs]


        connmatrix = torch.zeros((from_idxs.shape[0]+1, next_idxs.shape[0]+1), device=torch.device("cuda")).float()
        ## always set that pseudolabels match
        connmatrix[0, 0] = 1

        for i, c in enumerate(next_idxs):
            connmatrix[0, i+1] = self.connmatrix[0, c+1]
        for i, r in enumerate(from_idxs):
            connmatrix[i+1, 0] = self.connmatrix[r+1, 0]

        for i, r in enumerate(from_idxs):
            for j, c in enumerate(next_idxs):
                connmatrix[i+1, j+1] = self.connmatrix[r+1, c+1]


        ## some fruits may have been dropout from the original self.connmatrix
        ## we need to take this into consideration.
        for r in range(connmatrix.shape[0]):
            if connmatrix[r, :].sum()==0:
                connmatrix[r, 0] = 1

        for c in range(connmatrix.shape[1]):
            if connmatrix[:, c].sum()==0:
                connmatrix[0, c] = 1

        return next_centered, connmatrix

    def printSummary(self, data_idx):
        x_min = self.dataset_from.centers[:, 0].min()
        x_max = self.dataset_from.centers[:, 0].max()
        print(f"|** dataset {data_idx:2d}  - from {x_min:5.2f} to {x_max:5.2f}, contains {self.matched_num:4d} connections **|")
        print(f" -  from contains {self.dataset_from.getStrawberriesNum():4d} strawberries ({self.nomatch_from_num:4d} are not matched)")
        print(f" -  next contains {self.dataset_next.getStrawberriesNum():4d} strawberries ({self.nomatch_next_num:4d} are not matched)")
        #print(f"|**************************|")
        print()

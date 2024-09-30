import open3d as o3d
import json
from tqdm import tqdm
import yaml
import numpy as np
import open3d.visualization as vis
import random
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import glob, os
import MinkowskiEngine as ME


def transform(point, T):
    point = np.hstack([np.array(point), np.array([1])])
    return np.matmul(T, point)[:3]



class Strawberries(Dataset):
    def __init__(self, ply_path, annopath, T, min_x, max_x, min_x2=None, max_x2=None, training=False):
        self.centers = []
        self.failcenters = []
        self.failpath = []
        self.keys = []
        self.fruit_radius = []
        self.T = T
        self.foldername1 = ply_path #f"../strawberries_{seq}"
        self.foldername2 = ply_path + "_more" #f"../strawberries_{seq}_more"
        #self.foldername_detectorfail = f"../strawberries_detectorfailssimulation_{seq}"
        self.training = training

        self.nomink = False
        self.N_max = 500

        self.generatefails = False

        with open(annopath) as f:
            annotations = json.load(f)

        for k in annotations:
            self.centers.append(annotations[k]['center'])
            self.keys.append(annotations[k]['id'])
            self.fruit_radius.append(annotations[k]['radius'])

        self.centers = np.array(self.centers)
        self.keys    = np.array(self.keys)
        self.fruit_radius    = np.array(self.fruit_radius)

        #### transform centers to world frame ####
        pts = np.ones((self.centers.shape[0], 4))
        pts[:, :3] = self.centers
        pts = pts.T
        self.centers = np.matmul(self.T, pts).T[:, :3]

        #print("min x: ", self.centers[:, 0].min())
        #print("max x: ", self.centers[:, 0].max())

        if self.training and self.generatefails:

            files = glob.glob(f"{self.foldername_detectorfail}/fail_*")
            for file in files:
                pcd = o3d.io.read_point_cloud(file)
                self.failcenters.append(pcd.get_center())
                self.failpath.append(os.path.basename(file))
            
            self.failcenters = np.array(self.failcenters)
            self.failpath = np.array(self.failpath)


            #### transform centers to world frame ####
            pts = np.ones((self.failcenters.shape[0], 4))
            pts[:, :3] = self.failcenters
            pts = pts.T
            self.failcenters = np.matmul(self.T, pts).T[:, :3]


        #### load only part of the dataset ####
        if training:
            if min_x2 is None:
                mask  = np.logical_and(self.centers[:, 0]>min_x,  self.centers[:, 0]<max_x)
            elif min_x is None:
                mask  = np.logical_and(self.centers[:, 0]>min_x2, self.centers[:, 0]<max_x2)
            else:
                mask1 = np.logical_and(self.centers[:, 0]>min_x,  self.centers[:, 0]<max_x)
                mask2 = np.logical_and(self.centers[:, 0]>min_x2, self.centers[:, 0]<max_x2)
                mask  = np.logical_or(mask1, mask2)
        else:
            mask = np.logical_and(self.centers[:, 0]>min_x,self.centers[:, 0]<max_x)

        self.centers = self.centers[mask]
        self.keys    = self.keys[mask].tolist()
        self.fruit_radius    = self.fruit_radius[mask].tolist()

        if self.training and self.generatefails:

            mask = np.logical_and(self.failcenters[:, 0]>min_x,self.failcenters[:, 0]<max_x)
            self.failcenters = self.failcenters[mask]
            self.failkeys    = np.ones(self.failcenters.shape[0], dtype=int)*-1

            failidxs = np.arange(self.failcenters.shape[0])
            np.random.shuffle(failidxs)
            failidxs = failidxs[:10]
            self.failcenters = self.failcenters[failidxs]
            self.failpath    = self.failpath[failidxs]
            self.failkeys    = self.failkeys[failidxs]


        self.real_centers_num = self.centers.shape[0]

        if self.training and self.generatefails:
            self.centers = np.concatenate((self.centers, self.failcenters), axis=0)
            self.keys = np.concatenate((self.keys, self.failkeys), axis=0).tolist()

        

        #### compute relative position of centers ####
        self.relative_positions = []
        for i, center in enumerate(self.centers):
            self.relative_positions.append(self.centers - center)


        self.radius = 0.2
        self.res = 0.0005
    
    def __len__(self):
        return len(self.centers)

    def getStrawberriesNum(self):
        return self.centers.shape[0]


    def pc2img(self, path, coords, colors, lookat=None):
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=800, height=800, window_name="datasetgcn")
        visualizer.add_geometry(pcd)
        view_ctl = visualizer.get_view_control()
        view_ctl.set_up((0.11816762415842456, -0.27732487015278556, 0.95348168781340203))  # set the negative direction of the y-axis as the up direction
        view_ctl.set_front((-0.82126375901488036, -0.56704358549644407, -0.063145944256822337))  # set the positive direction of the x-axis toward you
        if lookat:
            view_ctl.set_lookat(lookat)  # set the original point as the center point of the window
        else:
            view_ctl.set_lookat((0.0052461686490607841, -0.0024268156850036182, 0.0038237402938542196))  # set the original point as the center point of the window
        view_ctl.set_zoom(0.86)
        visualizer.run()#update_renderer()
        quit()
        #visualizer.capture_screen_image(path)
        self.toalpha(path)

    def load(self, path, shift):
#        path = f"{self.foldername1}/straw_{str(key).zfill(3)}.ply"
#        pcd1 = o3d.io.read_point_cloud(path)
#        pcd1.transform(self.T)
#        if shift is not None:
#            pcd1.translate(shift)

        
        pcd = o3d.io.read_point_cloud(path)
        pcd.transform(self.T)
        if shift is not None:
            pcd.translate(shift)

        #### Data Augmentation ####
        if self.training:
            roll  = random.random() * 60 - 30.0
            pitch = random.random() * 60 - 30.0
            yaw   = random.random() * 60 - 30.0
            rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
            #pcd1.rotate(rot.as_matrix())
            pcd.rotate(rot.as_matrix())

            # jittering
            points = np.asarray(pcd.points)
            points += np.random.normal(0, 0.0007, points.shape)
            pcd.points = o3d.utility.Vector3dVector(points)

            colors  = np.asarray(pcd.colors)
            colors += np.random.normal(0, 0.05, colors.shape)
            colors  = np.clip(colors, 0, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            #self.pc2img("aa", np.asarray(pcd.points), colors)
            #quit()

        return pcd

    def get_concat(self, pcd):
        return np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)])

    def __getitem__(self, idx):

        key    = self.keys[idx]
        center = self.centers[idx]

        if idx<self.real_centers_num:
            path = f"{self.foldername2}/straw_{str(int(float(key))).zfill(3)}.ply"
        else:
            idx_o = idx - self.real_centers_num
            path = f"{self.foldername_detectorfail}/{self.failpath[idx_o]}"
        
        pcd = self.load(path, -center)

        pcd_coords = np.asarray(pcd.points)
        pcd_features = self.get_concat(pcd)

        if self.nomink:
            if pcd_features.shape[0]>self.N_max:
                idxs = np.arange(0, pcd_features.shape[0], 1)
                np.random.shuffle(idxs)
                idxs = idxs[:self.N_max]
                pcd_features = pcd_features[idxs]
            elif pcd_features.shape[0]<self.N_max:
                n = self.N_max - pcd_features.shape[0]
                tail = np.zeros((n, pcd_features.shape[1]))
                pcd_features = np.concatenate((pcd_features, tail))
            return idx, key, pcd_coords, pcd_features[None]

        return idx, key, pcd_coords, pcd_features


    def getO3dCloud(self, idx, shift):
        key    = self.keys[idx]
        path = f"{self.foldername1}/straw_{str(key).zfill(3)}.ply"
        return self.load(path, shift)


    def custom_collation_fn(self, data_labels):
        idxs_b, keys_b, pcds_coords_b, pcds_features_b = list(zip(*data_labels))

        
        if self.training:
            keep_number = round(self.keep_ratio * len(keys_b))
            #print(f"keep ratio {self.keep_ratio:4.2f} %  - {keep_number:4d} / {len(keys_b):4d}")
            
            idxs_b = idxs_b[:keep_number]
            keys_b = keys_b[:keep_number]
            pcds_coords_b   =   pcds_coords_b[:keep_number]
            pcds_features_b = pcds_features_b[:keep_number]


        if self.nomink:
            pcds_features_b  = np.concatenate(pcds_features_b, 0)
            pcds_features_b = torch.from_numpy(pcds_features_b).float().cuda().permute(0, 2, 1)

            return torch.Tensor(idxs_b).long(), pcds_features_b, keys_b

        voxelized_coords = [p / self.res for p in pcds_coords_b]
        feats  = np.concatenate(pcds_features_b, 0)

        #pcds = ME.TensorField(
            #features=torch.from_numpy(feats).float(),
            #coordinates=ME.utils.batched_coordinates(voxelized_coords, dtype=torch.float32),
            #quantization_mode=ME.SparseTensorQuantizationMode.
            #UNWEIGHTED_AVERAGE,
            #minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            #device="cuda",
        #)

        pcds = ME.TensorField(
            features=torch.from_numpy(feats).float(),
            coordinates=ME.utils.batched_coordinates(voxelized_coords, dtype=torch.float32),
            device="cuda",
        )

        return torch.Tensor(idxs_b).long(), pcds, keys_b
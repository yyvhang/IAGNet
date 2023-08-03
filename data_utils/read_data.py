import os
from os.path import join as opj
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import json
import pickle as pkl
import pdb

'''
load raw data from 3D-AffordanceNet dataset
'''


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def semi_points_transform(points):
    spatialExtent = np.max(points, axis=0) - np.min(points, axis=0)
    eps = 2e-3*spatialExtent[np.newaxis, :]
    jitter = eps*np.random.randn(points.shape[0], points.shape[1])
    points_ = points + jitter
    return points_


class AffordNetDataset(Dataset):
    def __init__(self, data_dir, split, partial=False, rotate='None', semi=False):
        super().__init__()
        #point_init
        self.data_dir = data_dir
        self.split = split

        self.partial = partial
        self.rotate = rotate
        self.semi = semi

        self.load_point_data()

        self.affordance = self.all_data[0]["affordance"]

        return

    def load_point_data(self):
        self.all_data = []
        if self.semi:
            with open(opj(self.data_dir, 'semi_label_1.pkl'), 'rb') as f:
                temp_data = pkl.load(f)
        else:
            if self.partial:
                with open(opj(self.data_dir, 'partial_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            elif self.rotate != "None" and self.split != 'train':
                with open(opj(self.data_dir, 'rotate_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data_rotate = pkl.load(f)
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
            else:
                with open(opj(self.data_dir, 'full_shape_%s_data.pkl' % self.split), 'rb') as f:
                    temp_data = pkl.load(f)
        for index, info in enumerate(temp_data):
            if self.partial:
                partial_info = info["partial"]
                for view, data_info in partial_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["view_id"] = view
                    temp_info["data_info"] = data_info
                    self.all_data.append(temp_info)
            elif self.split != 'train' and self.rotate != 'None':
                rotate_info = temp_data_rotate[index]["rotate"][self.rotate]
                full_shape_info = info["full_shape"]
                for r, r_data in rotate_info.items():
                    temp_info = {}
                    temp_info["shape_id"] = info["shape_id"]
                    temp_info["semantic class"] = info["semantic class"]
                    temp_info["affordance"] = info["affordance"]
                    temp_info["data_info"] = full_shape_info
                    temp_info["rotate_matrix"] = r_data.astype(np.float32)
                    self.all_data.append(temp_info)
            #only use the following:
            else:
                temp_info = {}
                temp_info["shape_id"] = info["shape_id"]
                temp_info["semantic class"] = info["semantic class"]
                temp_info["affordance"] = info["affordance"]
                temp_info["data_info"] = info["full_shape"]
                self.all_data.append(temp_info)

    def __getitem__(self, index):

        data_dict = self.all_data[index]
        modelid = data_dict["shape_id"]   #object id
        modelcat = data_dict["semantic class"]  #object class
        affordance_list = data_dict["affordance"]
        data_info = data_dict["data_info"]
        model_data = data_info["coordinate"].astype(np.float32)
        labels = data_info["label"]
        for aff in self.affordance:
            temp = labels[aff].astype(np.float32).reshape(-1, 1)
            model_data = np.concatenate((model_data, temp), axis=1)

        datas = model_data[:, :3]
        targets = model_data[:, 3:]
        datas, _, _ = pc_normalize(datas)

        return datas, targets, affordance_list, modelid, modelcat

    def __len__(self):
        return len(self.all_data)

if __name__=='__main__':
    path = 'Research/Code_project/full-shape'
    val_dataset = AffordNetDataset(path,'val', partial=False, rotate='None', semi=False)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4, drop_last=False)
    m = torch.FloatTensor([0])

    all_affordance = []
    for idx, (data, label,affordance,  _, _) in enumerate(val_loader):
        x = data  
        y = label 
        print(affordance)
    

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from PIL import Image
from torchvision import transforms
import pdb
import random

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class Point_dataset(Dataset):
    def __init__(self, img_path, point_path, run_type):
        super().__init__()

        self.p_path = point_path  #txt: all file path
        self.i_path = img_path
        self.point_files = self.read_file(self.p_path)
        self.img_files = self.read_file(self.i_path)
        self.run_type = run_type

        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']


        self.object_list = ['Knife', 'Refrigerator', 'Earphone', 
            'Bag', 'Keyboard', 'Chair', 'Hat', 'Door', 'TrashCan', 'Table', 
            'Faucet', 'StorageFurniture', 'Bottle', 'Bowl', 'Display', 'Mug', 'Clock']
        self.object_train_split = [[0, 272], [272, 430], [430, 622], 
            [622, 725], [725, 860], [860, 2411], [2411, 2602], [2602, 2735], [2735, 2984], 
            [2984, 4041], [4041, 4299], [4299, 4681], [4681, 5030], [5030, 5190], [5190, 5495], [5495, 5682], [5682, 5896]]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        object_name = img_path.split('_')[-3]

        if (self.run_type=='val'):
            point_path = self.point_files[index]
        else:
            idx = self.object_list.index(object_name)
            range_ = self.object_train_split[idx]
            point_sample_idx = random.sample(range(range_[0],range_[1]),1)
            point_path = self.point_files[point_sample_idx[0]]

        Point, affordance_label = self.extract_point_file(point_path)

        Point,_,_ = pc_normalize(Point)
        Point = Point.transpose()
        affordance_label = self.get_affordance_label(img_path, affordance_label)
        return Point, affordance_label

    def read_file(self, path):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                file_list.append(file)

            f.close()
        return file_list
    
    def extract_point_file(self, path):
        with open(path,'r') as f:
            coordinates = []
            lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            line = line.strip(' ')
            data = line.split(' ')
            coordinate = [float(x) for x in data[2:]]
            coordinates.append(coordinate)
        data_array = np.array(coordinates)
        points_coordinates = data_array[:, 0:3]
        affordance_label = data_array[: , 3:]

        return points_coordinates, affordance_label

    def get_affordance_label(self, str, label):
        cut_str = str.split('_')
        affordance = cut_str[-2]
        index = self.affordance_label_list.index(affordance)

        label = label[:, index]
        
        return label

if __name__=='__main__':
    point_train_path = 'Data/Point_Train.txt'
    train_dataset = Point_dataset(point_train_path)
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=8, shuffle=True)
    print(f'len_dataset:{len(train_dataset)}')
    print(f'len_dataloader:{len(train_loader)}')
    for i,(point, label) in enumerate(train_loader):
        '''
        img : [B, C, H, W]
        point: [B, 3, N]
        label: [B, N, 18]
        '''
  









import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
import pdb
import json
import random

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def img_normalize_train(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def img_normalize_val(img, scale=256/224, input_size=224):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

class PIAD(Dataset):
    def __init__(self, run_type, setting_type, point_path, img_path, box_path, pair=2, img_size=(224, 224)):
        super().__init__()

        self.run_type = run_type
        self.p_path = point_path
        self.i_path = img_path
        self.b_path = box_path
        self.pair_num = pair
        self.affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab']
        '''
        Unseen
        '''
        if setting_type == 'Unseen':
            number_dict = {'Knife': 0, 'Refrigerator': 0, 'Earphone': 0, 
            'Bag': 0, 'Keyboard': 0, 'Chair': 0, 'Hat': 0, 'Door': 0, 'TrashCan': 0, 'Table': 0, 
            'Faucet': 0, 'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Display': 0, 'Mug': 0, 'Clock': 0}

        '''
        Seen
        '''
        if setting_type == 'Seen':
            number_dict = {'Earphone': 0, 'Bag': 0, 'Chair': 0, 'Refrigerator': 0, 'Knife': 0, 'Dishwasher': 0, 'Keyboard': 0, 'Scissors': 0, 'Table': 0, 
            'StorageFurniture': 0, 'Bottle': 0, 'Bowl': 0, 'Microwave': 0, 'Display': 0, 'TrashCan': 0, 'Hat': 0, 'Clock': 0, 
            'Door': 0, 'Mug': 0, 'Faucet': 0, 'Vase': 0, 'Laptop': 0, 'Bed': 0}

        self.img_files = self.read_file(self.i_path)
        self.box_files = self.read_file(self.b_path)
        self.img_size = img_size

        if self.run_type == 'train':
            self.point_files, self.number_dict = self.read_file(self.p_path, number_dict)
            self.object_list = list(number_dict.keys())
            self.object_train_split = {}
            start_index = 0
            for obj_ in self.object_list:
                temp_split = [start_index, start_index + self.number_dict[obj_]]
                self.object_train_split[obj_] = temp_split
                start_index += self.number_dict[obj_]
        else:
            self.point_files = self.read_file(self.p_path)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        img_path = self.img_files[index]
        box_path = self.box_files[index]

        if (self.run_type=='val'):
            point_path = self.point_files[index]
        else:
            object_name = img_path.split('_')[-3]
            range_ = self.object_train_split[object_name]
            point_sample_idx = random.sample(range(range_[0],range_[1]), self.pair_num)

        Img = Image.open(img_path).convert('RGB')

        if(self.run_type == 'train'):
            Img, subject, object = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Points_List = []
            affordance_label_List = []
            affordance_index_List = []
            for id_x in point_sample_idx:
                point_path = self.point_files[id_x]
                Points, affordance_label = self.extract_point_file(point_path)
                Points,_,_ = pc_normalize(Points)
                Points = Points.transpose()
                affordance_label, affordance_index = self.get_affordance_label(img_path, affordance_label)
                Points_List.append(Points)
                affordance_label_List.append(affordance_label)
                affordance_index_List.append(affordance_index)

        else:
            subject, object = self.get_crop(box_path, Img, self.run_type)
            sub_box, obj_box = self.get_resize_box(Img, self.img_size, subject, object)
            sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()
            Img = Img.resize(self.img_size)
            Img = img_normalize_train(Img)

            Point, affordance_label = self.extract_point_file(point_path)
            Point,_,_ = pc_normalize(Point)
            Point = Point.transpose()

            affordance_label,_ = self.get_affordance_label(img_path, affordance_label)

        if(self.run_type == 'train'):
            return Img, Points_List, affordance_label_List, affordance_index_List, sub_box, obj_box
        else:
            return Img, Point, affordance_label, img_path, point_path, sub_box, obj_box

    def read_file(self, path, number_dict=None):
        file_list = []
        with open(path,'r') as f:
            files = f.readlines()
            for file in files:
                file = file.strip('\n')
                if number_dict != None:
                    object_ = file.split('_')[-2]
                    number_dict[object_] +=1
                file_list.append(file)

            f.close()
        if number_dict != None:
            return file_list, number_dict
        else:
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
        
        return label, index

    def get_crop(self, json_path, image, run_type):

        json_data = json.load(open(json_path, 'r'))
        sub_points, obj_points = [], []
        for box in json_data['shapes']:
            if(box['label'] == 'subject'):
                sub_points = box['points']
            elif(box['label'] == 'object'):
                obj_points = box['points']
        if(len(sub_points) == 0):
            temp_box = [0.] * 2
            for i in range(2):
                sub_points.append(temp_box)
        
        if(run_type=='train'):
            crop_img, crop_subpoints, crop_objpoints = self.random_crop_with_points(image, sub_points, obj_points)
            return crop_img, crop_subpoints, crop_objpoints
        else:
            sub_points = [*sub_points[0], *sub_points[1]]
            obj_points = [*obj_points[0], *obj_points[1]]
            sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
            return sub_points, obj_points

    def random_crop_with_points(self, image, sub_points, obj_points):

        points = []
        image = np.array(image)
        for obj_point in obj_points:
            points.append(obj_point)

        for sub_point in sub_points:
            points.append(sub_point)

        h, w = image.shape[0], image.shape[1]
        points = np.array(points, np.int32)
        min_x, min_y, max_x, max_y = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])

        t, b, lft, r = (random.randint(0, min_y),
                        random.randint(max_y + 1, h) if max_y + 1 < h else max_y + 1,
                        random.randint(0, min_x),
                        random.randint(max_x + 1, w) if max_x + 1 < w else max_x + 1)

        new_img = image[t: b, lft: r, :]

        new_img = Image.fromarray(new_img)
        obj_points = points[0:2]
        new_objpoints = [[x - lft, y - t] for x, y in obj_points]
        obj_LT = new_objpoints[0]
        obj_RB = new_objpoints[1]
        new_objpoints = [*obj_LT, *obj_RB]   #[x1, y1, x2, y2] left_top & right_bottom

        sub_points = points[2:]
        new_subpoints = [[x - lft, y - t] for x, y in sub_points]
        sub_LT = new_subpoints[0]
        sub_RB = new_subpoints[1]
        new_subpoints = [*sub_LT, *sub_RB]   #[x1, y1, x2, y2] left_top & right_bottom

        return new_img, new_subpoints, new_objpoints

    def get_resize_box(self, Image, new_size, sub_box, obj_box):

        Image = np.array(Image)
        h_ = Image.shape[0]
        w_ = Image.shape[1]

        scale_h = new_size[0] / h_
        scale_w = new_size[1] / w_


        sub_box[0], sub_box[2] = sub_box[0] * scale_w, sub_box[2] * scale_w
        sub_box[1], sub_box[3] = sub_box[1] * scale_h, sub_box[3] * scale_h

        obj_box[0], obj_box[2] = obj_box[0] * scale_w, obj_box[2] * scale_w
        obj_box[1], obj_box[3] = obj_box[1] * scale_h, obj_box[3] * scale_h

        return sub_box, obj_box
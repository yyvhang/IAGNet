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


def get_crop(json_path, image, run_type):

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
        crop_img, crop_subpoints, crop_objpoints = random_crop_with_points(image, sub_points, obj_points)
        return crop_img, crop_subpoints, crop_objpoints
    else:
        sub_points = [*sub_points[0], *sub_points[1]]
        obj_points = [*obj_points[0], *obj_points[1]]
        sub_points, obj_points = np.array(sub_points, np.int32), np.array(obj_points, np.int32)
        return sub_points, obj_points

def random_crop_with_points(image, sub_points, obj_points):

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

def get_resize_box(Image, new_size, sub_box, obj_box):

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

import torch
import argparse
import numpy as np
import pdb
import os
from PIL import Image
from model.IAGNet import get_IAGNet
from utils.visualization_point import get_affordance_label
from torchvision import transforms
import open3d as o3d
from utils.get_box import get_crop, get_resize_box

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def img_normalize(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

def extract_point_file(point_path):
    with open(point_path,'r') as f:
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

    return points_coordinates

def inference_single(img_path, box_path, GT_path, model_path, results_folder):

    model = get_IAGNet(pre_train=False)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    model.eval()

    Img = Image.open(img_path).convert('RGB')
    subject, object = get_crop(box_path, Img, 'inference')
    sub_box, obj_box = get_resize_box(Img, (224, 224), subject, object)
    sub_box, obj_box = torch.tensor(sub_box).float(), torch.tensor(obj_box).float()

    Img = Img.resize((224, 224))
    Img = img_normalize(Img)
    Img = Img.unsqueeze(0).cuda()
    sub_box = sub_box.unsqueeze(0).cuda()
    obj_box = obj_box.unsqueeze(0).cuda()

    with open(GT_path,'r') as f:
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
        affordance_label = get_affordance_label(img_path, affordance_label)

        Point = pc_normalize(points_coordinates)
        Point = Point.transpose()
        Points = torch.from_numpy(Point)
        Points = torch.unsqueeze(Points, 0)
        Points = Points.float().cuda()
        pred,_,_ = model(Img, Points, sub_box, obj_box)
        pred = torch.squeeze(pred)
        affordance_pred = pred.cpu().detach().numpy()

        gt_point = o3d.geometry.PointCloud()
        gt_point.points = o3d.utility.Vector3dVector(points_coordinates)

        pred_point = o3d.geometry.PointCloud()
        pred_point.points = o3d.utility.Vector3dVector(points_coordinates)

        color = np.zeros((2048,3))
        reference_color = np.array([255, 0, 0])
        back_color = np.array([190, 190, 190])

        for i, point_affordacne in enumerate(affordance_label):
            scale_i = point_affordacne
            color[i] = (reference_color-back_color) * scale_i + back_color
        gt_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        pred_color = np.zeros((2048,3))

        for i, aff_pred in enumerate(affordance_pred):
            scale_i = aff_pred
            pred_color[i] = (reference_color-back_color) * scale_i + back_color
        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
        pred_point.translate((2, 0, 0), relative=True)

        object = GT_path.split('_')[-2]
        affordance_type = img_path.split('_')[-2]
        num = (GT_path.split('_')[-1]).split('.')[0]
        GT_file = results_folder + object + '_' + affordance_type + '_' + num + '_GT' + '.ply'
        pred_file = results_folder + object + '_' + affordance_type + '_' + num + '_Pred' + '.ply'

        o3d.visualization.draw_geometries([gt_point, pred_point], window_name='GT point', width=600, height=600)

        o3d.io.write_point_cloud(pred_file, pred_point)
        o3d.io.write_point_cloud(GT_file, gt_point)
        f.close()    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='ckpts/IAG_Seen.pt', help='model path')
    parser.add_argument('--img_path', type=str, default='Demo/Img_Test_Bag_lift_1.jpg', help='test img path')
    parser.add_argument('--point_path', type=str, default='Demo/Point_Test_Bag_1.txt', help='test point path')
    parser.add_argument('--box_path', type=str, default='Demo/Img_Test_Bag_lift_1.json', help='test point path')
    parser.add_argument('--results_path', type=str, default='Demo/', help='save Demo path')

    opt = parser.parse_args()

    inference_single(opt.img_path, opt.box_path, opt.point_path, opt.model_path, opt.results_path)


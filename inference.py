
import torch
import argparse
import numpy as np
import pdb
import os
import threading
import seaborn as sns
from PIL import Image
from model.pn2 import PointNet_Estimation
from model.IAGNet import get_IAGNet
from utils.visualization_point import visual_pred, get_affordance_label, visual_pointpred
from torchvision import transforms
from data_utils.dataset_PIAD import PIAD
from data_utils.dataset_point import Point_dataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from multiprocessing import Process
import matplotlib.pyplot as plt
import open3d as o3d
from open3d.visualization import O3DVisualizer
import open3d.visualization.gui as gui
from utils.get_box import get_crop, get_resize_box
import pandas as pd

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

def inference_single(img_path, GT_path, results_folder, model_path, box_path):

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

def visual_single(img_path, affordance_pred, GT_path, results_folder, model_path):

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
        for i in range(pred_color.shape[0]):
            pred_color[i] = back_color


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

def inference_pointpred(opt):
    model = PointNet_Estimation(num_classes=1)
    checkpoint = torch.load(opt.model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    dataset = Point_dataset(opt.batch_img_path, opt.batch_point_path, 'val')
    dataloder = DataLoader(dataset, batch_size=16, num_workers=8)

    with open(opt.batch_img_path, 'r') as f:
        img_files = f.readlines()
        f.close()
    for i,(point, label) in enumerate(dataloder):

        point = point.float().cuda()

        pred = model(point)

        num = pred.shape[0]
        pred = pred.cpu()
        pred = pred.detach().numpy()
        point = point.cpu()
        point = point.detach().numpy()
        label = label.numpy()
        for j in range(num):
            img_path = img_files[i*16+j]
            coords = point[j].transpose(1,0)
            visual_pointpred(img_path, coords, pred[j], label[j], opt.results_path)

def inference_batch(opt):

    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    model = get_IAGNet(pre_train=False)
    checkpoint = torch.load(opt.model_path, map_location='cuda:0')

    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    dataset = PIAD('val', opt.visual_type, opt.batch_point_path, opt.batch_img_path, opt.batch_box_path)
    dataloder = DataLoader(dataset, batch_size=16, num_workers=8)

    for i,(img, point, _, img_file, point_file,sub_box, obj_box) in enumerate(dataloder):
        point = point.float()
        img, point = img.cuda(), point.cuda()
        sub_box, obj_box = sub_box.cuda(), obj_box.cuda()

        pred, aff_pred, simliarity = model(img, point, sub_box, obj_box)
        
        num = pred.shape[0]
        pred = pred.cpu()
        pred = pred.detach().numpy()
        for i in range(num):
            visual_pred(img_file[i], pred[i], point_file[i], opt.results_path)

    print('Finsh!')

def visual_batch(path):
    file_list = os.listdir(path)
    file_list.sort()
    start = 0
    while(start < len(file_list)):
        pair = file_list[start : start+2]
        pair[0] = path + pair[0]
        pair[1] = path + pair[1]

        model_name = path.split('/')[2]
        name = model_name + ' | ' + pair[0].split('/')[-1]
        gt = o3d.io.read_point_cloud(pair[0])
        pred = o3d.io.read_point_cloud(pair[1])
        o3d.visualization.draw_geometries([gt, pred], width=600, height=600, window_name=name)
        start += 2

def visual_single_ply(path, ply, GT):

    pred_path = path + ply
    GT_path = path + GT
    model_name = path.split('/')[2]
    name = model_name + ' | ' + ply
    gt = o3d.io.read_point_cloud(GT_path)
    pred = o3d.io.read_point_cloud(pred_path)
    o3d.visualization.draw_geometries([gt, pred], width=600, height=600, window_name=name)
    start += 2

def multi_window_visual():
    Processes = [
        Process(target=visual_single_ply, args=('Data/Results/IAG/',)),
        Process(target=visual_single_ply, args=('Data/Results/IAG_Unseen/',)),
    ]
    for process in Processes:
        process.start()

def multi_single_visual(opt):
    Processes = [
        Process(target=visual_single_ply, args=('Data/Results/IAG/',opt.single_ply, opt.single_plygt)),
        Process(target=visual_single_ply, args=('Data/Results/IAG_Unseen/', opt.single_ply, opt.single_plygt)),
    ]
    for process in Processes:
        process.start()

def t_SNE(opt):
    model = get_IAGNet(pre_train=False)
    checkpoint = torch.load(opt.model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model.cuda()
    model.eval()

    dataset = PIAD('val', opt.visual_type, opt.batch_point_path, opt.batch_img_path, opt.batch_box_path)
    dataloder = DataLoader(dataset, batch_size=16, num_workers=8)
    Label = []
    Pred = []
    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(2,4))
    with torch.no_grad():
        for i,(img, point, _, img_file, point_file,aff_type,_,sub_box, obj_box) in enumerate(dataloder):
            point = point.float()
            img, point = img.cuda(), point.cuda()
            sub_box, obj_box = sub_box.cuda(), obj_box.cuda()
            _, aff_pred, _ = model(img, point, sub_box, obj_box)
            _,pred_index = torch.max(aff_pred,1)

            for cls_index in pred_index:
                Label.append(int(cls_index))

            print(f'iter:{i}')
            if(i==0):
                Pred = aff_pred.detach().cpu().numpy()
            else:
                Pred = np.concatenate((Pred, aff_pred.detach().cpu().numpy()),axis=0)
        n_components = 2
        ts = TSNE(n_components=n_components, init='pca', random_state=0, early_exaggeration=20,n_iter=2000, perplexity=20)
        y = ts.fit_transform(Pred)
        y_min,y_max = y.min(0),y.max(0)
        y_norm = (y-y_min) / (y_max - y_min)

        color_label = np.array(Label)

        cm = 'Set1'
        ax_1.scatter(y_norm[:,0],y_norm[:,1], c=color_label, cmap=cm, s=2, alpha=0.5, linewidths=0.0)
        ax_1.axis('off')
        fig.savefig('Data/Results/tSNE.jpg',bbox_inches='tight', pad_inches = 0, dpi=800)    

def get_raw(path):

    raw_path = 'Data/Chair_sit_GT.ply'
    GT = o3d.io.read_point_cloud(path)
    coords = np.asarray(GT.points)

    back_color = np.array([190, 190, 190])
    raw_point = o3d.geometry.PointCloud()
    raw_point.points = o3d.utility.Vector3dVector(coords)
    color = np.zeros((2048,3))
    for i in range(color.shape[0]):
        color[i] = back_color
    raw_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(raw_path, raw_point)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='runs/train/IAG_UnseenT/Epoch_26.pt', help='model path')
    parser.add_argument('--img_path', type=str, default='Data/test.jpg', help='test img path')
    parser.add_argument('--point_path', type=str, default='Data/test.txt', help='test point path')
    parser.add_argument('--box_path', type=str, default='Data/test.json', help='test point path')

    parser.add_argument('--batch_img_path', type=str, default='Data/Seen/Img_Test.txt', help='batch img path')
    parser.add_argument('--batch_point_path', type=str, default='Data/Seen/Point_Test.txt', help='batch point path')
    parser.add_argument('--batch_box_path', type=str, default='Data/Seen/Box_Test.txt', help='batch box path')

    parser.add_argument('--results_path', type=str, default='runs/train/IAG_UnseenT/visual_IAG/', help='batch img path')
    parser.add_argument('--visual_type', type=str, default='Unseen', help='Seen or Unseen or Unseen_2')
    parser.add_argument('--point_folder', type=str, default='Data/Results/IAG/', help='Point folder')
    opt = parser.parse_args()


    inference_batch(opt)
    #visual_batch(opt.results_path)


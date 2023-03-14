import pickle as pkl
import os
import pdb
from turtle import width
from matplotlib.pyplot import axis
import numpy as np
import open3d as o3d


Affordance_label_list = ['grasp', 'contain', 'lift', 'open', 
                'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                'push', 'pull', 'listen', 'wear', 'press', 'cut', 'stab']
color_list = [[252, 19, 19], [249, 113, 45], [247, 183, 55], [251, 251, 11], [178, 244, 44], [255, 0, 0], 
              [0, 0, 255], [25, 248, 99], [46, 253, 184], [40, 253, 253], [27, 178, 253], [28, 100, 243], 
              [46, 46, 125], [105, 33, 247], [172, 10, 253], [249, 47, 249], [253, 51, 186], [250, 18, 95]]
color_list = np.array(color_list)
def get_affordance_label(str, label):
    cut_str = str.split('_')
    affordance = cut_str[-2]
    index = Affordance_label_list.index(affordance)

    label = label[:, index]
    
    return label
'''
use txt dataformat
'''
def visual_txt(Txt_path):
    with open(Txt_path,'r') as f:
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
        visual_point = o3d.geometry.PointCloud()
        visual_point.points = o3d.utility.Vector3dVector(points_coordinates)

        color = np.zeros((2048,3))
        for i,point_affordance in enumerate(affordance_label):
            if(np.max(point_affordance) > 0):
                color_index = np.argmax(point_affordance)
                color[i] = color_list[color_index]

        visual_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

        o3d.visualization.draw_geometries([visual_point])
        f.close()

def visual_pred(img_path, affordance_pred, GT_path, results_folder):
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
        for i, pred in enumerate(affordance_pred):

            scale_i = pred
            pred_color[i] = (reference_color-back_color) * scale_i + back_color

        pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
        pred_point.translate((2, 0, 0), relative=True)
        object = GT_path.split('_')[-2]
        affordance_type = img_path.split('_')[-2]
        num = (GT_path.split('_')[-1]).split('.')[0]
        GT_file = results_folder + object + '_' + affordance_type + '_' + num + '_GT' + '.ply'
        pred_file = results_folder + object + '_' + affordance_type + '_' + num + '_Pred' + '.ply'
        o3d.io.write_point_cloud(pred_file, pred_point)
        o3d.io.write_point_cloud(GT_file, gt_point)
        f.close()

'''
 use pkl dataformat
'''
def visual_pkl(pkl_path):
    points_file = open(pkl_path, 'rb')
    temp_data = pkl.load(points_file)
    for index, info in enumerate(temp_data):
        shape_id = info['shape_id']
        affordance_label_list = info['affordance']
        object_class = info['semantic class']  
        points_coordinate = info['full_shape']['coordinate']
        affordance_label = info['full_shape']['label']
        Points_data = points_coordinate

        visual_point = o3d.geometry.PointCloud()
        visual_point.points = o3d.utility.Vector3dVector(Points_data)
        R = visual_point.get_rotation_matrix_from_xyz((np.pi/2,0,np.pi/4))
        visual_point.rotate(R)
        color = np.random.random((2048, 3))
        visual_point.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([visual_point])

    for aff in affordance_label_list:
        temp = affordance_label[aff].astype(np.float32).reshape(-1, 1)
        pdb.set_trace()
        Points_data = np.concatenate((Points_data, temp), axis=1)

def visual_pointpred(img_path, points, affordance_pred, affordance_label, results_folder):

    gt_point = o3d.geometry.PointCloud()
    gt_point.points = o3d.utility.Vector3dVector(points)

    pred_point = o3d.geometry.PointCloud()
    pred_point.points = o3d.utility.Vector3dVector(points)

    color = np.zeros((2048,3))
    reference_color = np.array([255, 0, 0])
    back_color = np.array([190, 190, 190])

    for i, label in enumerate(affordance_label):
        scale_i = label
        color[i] = (reference_color-back_color) * scale_i + back_color
    gt_point.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

    pred_color = np.zeros((2048,3))
    for i, pred in enumerate(affordance_pred):
        scale_i = pred
        pred_color[i] = (reference_color-back_color) * scale_i + back_color

    pred_point.colors = o3d.utility.Vector3dVector(pred_color.astype(np.float64) / 255.0)
    pred_point.translate((2, 0, 0), relative=True)

    object = img_path.split('_')[-3]
    affordance_type = img_path.split('_')[-2]
    num = (img_path.split('_')[-1]).split('.')[0]
    GT_file = results_folder + object + '_' + affordance_type + '_' + num + '_GT' + '.ply'
    pred_file = results_folder + object + '_' + affordance_type + '_' + num + '_Pred' + '.ply'
    o3d.io.write_point_cloud(pred_file, pred_point)
    o3d.io.write_point_cloud(GT_file, gt_point)
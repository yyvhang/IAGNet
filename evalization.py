from re import A
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from data_utils.dataset_PIAD import PIAD
from model.IAGNet import get_IAGNet
from utils.eval import evaluating, SIM
from numpy import nan
import numpy as np
import pdb
import random
import os
import pandas as pd
import yaml

def Evalization(dataset, data_loader, model_path, use_gpu, Seeting):
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if(Seeting == 'Unseen_2'):
        object_list = ['Bottle', 'Bowl', 'Bed', 'Bag', 'Display', 'Dishwasher','Knife','Microwave','Mug','Scissors','Vase']
        Affordance_list = ['contain', 'lay', 'pour', 'wrapgrasp','open','display','stab','grasp']
    elif(Seeting == 'Unseen'):
        object_list = ['Bed', 'Dishwasher','Microwave','Scissors','Vase', 'Laptop']
        Affordance_list = ['contain', 'lay', 'sit', 'wrapgrasp','open','display','stab','grasp', 'press','cut']
    else:
        object_list = ['Vase', 'Display', 'Bed', 'Microwave', 'Door', 
        'Earphone', 'Bottle', 'Bowl', 'Laptop', 'Clock', 'Scissors', 'Mug', 'Faucet', 
        'StorageFurniture', 'Bag', 'Chair', 'Dishwasher', 'Refrigerator', 
        'Table', 'Hat', 'Keyboard', 'Knife', 'TrashCan']

        Affordance_list = ['grasp', 'contain', 'lift', 'open', 
                        'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                        'push', 'listen', 'wear', 'press', 'cut', 'stab']

    for obj in object_list:
        exec(f'{obj} = [[], [], [], []]')
    for aff in Affordance_list:
        exec(f'{aff} = [[], [], [], []]')

    model = get_IAGNet(pre_train=False)

    checkpoint = torch.load(model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    results = torch.zeros((len(dataset), 2048, 1))
    targets = torch.zeros((len(dataset), 2048, 1))
    total_point = 0
    num = 0
    with torch.no_grad():
        model.eval()
        Object = []
        Affordance = []
        for i,(img, point, label, img_path, point_path, sub_box, obj_box) in enumerate(data_loader):
            print(f'iteration: {i} start----')
            B = img.shape[0]
            for iter in range(B):
                object_class = point_path[iter].split('_')[-2]
                affordance_cls = img_path[iter].split('_')[-2]
                Object.append(object_class)
                Affordance.append(affordance_cls)

            point, label = point.float(), label.float()
            label = label.unsqueeze(dim=-1)
            if(use_gpu):
                img = img.to(device)
                point = point.to(device)
                label = label.to(device)
                sub_box, obj_box = sub_box.to(device), obj_box.to(device)

            pred,_,_ = model(img, point, sub_box, obj_box)

            pred_num = pred.shape[0]
            print(f'num:{num}, pred_num:{pred_num}')
            results[num : num+pred_num, :, :] = pred
            targets[num : num+pred_num, :, :] = label
            num += pred_num

        results = results.detach().numpy()
        targets = targets.detach().numpy()
        SIM_matrix = np.zeros(targets.shape[0])
        MAE_martrix = np.zeros(targets.shape[0])
        for i in range(targets.shape[0]):
            Sim = SIM(results[i], targets[i])
            mAE = np.sum(np.absolute(results[i]-targets[i])) / 2048
            SIM_matrix[i] = Sim
            MAE_martrix[i] = mAE
            object_cls = Object[i]
            aff_cls = Affordance[i]
            exec(f'{object_cls}[1].append({Sim})')
            exec(f'{aff_cls}[1].append({Sim})')
            exec(f'{object_cls}[3].append({mAE})')
            exec(f'{aff_cls}[3].append({mAE})')


        sim = np.mean(SIM_matrix)
        mean_MAE = np.mean(MAE_martrix)
        AUC = np.zeros((targets.shape[0], targets.shape[2]))
        IOU = np.zeros((targets.shape[0], targets.shape[2]))
        IOU_thres = np.linspace(0, 1, 20)
        targets = targets >= 0.5
        targets = targets.astype(int)
        for i in range(AUC.shape[0]):
            t_true = targets[i]
            p_score = results[i]
            object_cls = Object[i]
            aff_cls = Affordance[i]
            if np.sum(t_true) == 0:
                AUC[i] = np.nan
                IOU[i] = np.nan
                obj_auc = AUC[i]
                aff_auc = AUC[i]
                obj_iou = IOU[i]
                aff_iou = IOU[i]
                exec(f'{object_cls}[2].append({obj_auc})')
                exec(f'{aff_cls}[2].append({aff_auc})')
                exec(f'{object_cls}[0].append({obj_iou})')
                exec(f'{aff_cls}[0].append({aff_iou})')
            else:
                auc = roc_auc_score(t_true, p_score)
                AUC[i] = auc

                p_mask = (p_score > 0.5).astype(int)
                temp_iou = []
                for thre in IOU_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1.*intersect/union)
                temp_iou = np.array(temp_iou)
                aiou = np.mean(temp_iou)
                IOU[i] = aiou

                obj_auc = AUC[i]
                aff_auc = AUC[i]
                obj_iou = IOU[i]
                aff_iou = IOU[i]
                exec(f'{object_cls}[2].append({obj_auc})')   
                exec(f'{aff_cls}[2].append({aff_auc})')
                exec(f'{object_cls}[0].append({obj_iou})')
                exec(f'{aff_cls}[0].append({aff_iou})')

        AUC = np.nanmean(AUC)
        IOU = np.nanmean(IOU)
        print('------Object-------')
        for obj in object_list:
            aiou = np.nanmean(eval(obj)[0])
            sim_ = np.mean(eval(obj)[1])
            auc_ = np.nanmean(eval(obj)[2])
            mae_ = np.mean(eval(obj)[3])
            print(f'{obj} | IOU:{aiou} | SIM:{sim_} | AUC:{auc_}')

        avg_mertics = [0, 0, 0, 0]
        print('------Affordance-------')
        for i,aff in enumerate(Affordance_list):
            aiou = np.nanmean(eval(aff)[0])*100
            sim_ = np.mean(eval(aff)[1])
            auc_ = np.nanmean(eval(aff)[2])*100
            mae_ = np.mean(eval(aff)[3])
            avg_mertics[0] += aiou
            avg_mertics[1] += sim_
            avg_mertics[2] += auc_
            avg_mertics[3] += mae_
   
            print(f'{aff} | IOU:{aiou} | SIM:{sim_} | AUC:{auc_}, MAE:{mae_}')

        num_affordance = len(Affordance_list)
        avg_iou, avg_sim = avg_mertics[0] / num_affordance, avg_mertics[1] / num_affordance
        avg_auc, avg_mae = avg_mertics[2] / num_affordance, avg_mertics[3] / num_affordance

        print('------ALL-------')
        print(f'Overall---AUC:{AUC*100} | IOU:{IOU*100} | SIM:{sim} | MAE:{mean_MAE}')

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

if __name__=='__main__':

    dict = read_yaml('config/config_seen.yaml')
    point_path = dict['point_test']
    img_path = dict['img_test']
    box_path = dict['box_test']

    model_path = 'runs/train/IAG/best.pt'

    val_dataset = PIAD('val', dict['Setting'], point_path, img_path, box_path)
    val_loader = DataLoader(val_dataset, dict['batch_size'], num_workers=8, shuffle=True)

    use_gpu = True
    seed_torch(42)

    Evalization(val_dataset, val_loader, model_path, use_gpu, Seeting=dict['Setting'])

import argparse
from ast import parse
from ctypes import Union
from multiprocessing import reduction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils.dataset_PIAD import PIAD
from model.IAGNet import get_IAGNet 
from utils.loss import HM_Loss, kl_div
from utils.eval import evaluating, SIM
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from utils.Point_aug import random_scale_point_cloud, shift_point_cloud
import numpy as np
import os
import pdb
import logging
import random
import yaml

def read_yaml(path):
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def main(opt, dict):

    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)

    save_path = opt.save_dir + opt.name
    foler = os.path.exists(save_path)
    if not foler:
        os.makedirs(save_path)

    loger = logging.getLogger('Training')
    log_name = opt.save_dir + opt.name + '/' + opt.log_name
    logging.basicConfig(filename=log_name, level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARNING)
    def log_string(str):
        loger.info(str)
        print(str)


    img_train_path = dict['img_train']
    point_train_path = dict['point_train']
    img_val_path = dict['img_test']
    point_val_path = dict['point_test']
    box_train_path = dict['box_train']
    box_val_path = dict['box_test']
    Setting = dict['Setting']
    batch_size = dict['batch_size']

    log_string('Start loading train data---')
    train_dataset = PIAD('train', Setting, point_train_path, img_train_path, box_train_path, dict['pairing_num'])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8 , sampler=train_sampler, drop_last=True)
    log_string(f'train data loading finish, loading data files:{len(train_dataset)}')

    log_string('Start loading val data---')
    val_dataset = PIAD('val', Setting, point_val_path, img_val_path, box_val_path)
    test_num = len(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
    log_string(f'val data loading finish, loading data files:{len(val_dataset)}')


    # model, optimizer, loss
    model = get_IAGNet(img_model_path=dict['res18_pre'], N_p=dict['N_p'], emb_dim=dict['emb_dim'],
                       proj_dim=dict['proj_dim'], num_heads=dict['num_heads'], N_raw=dict['N_raw'],
                       num_affordance = dict['num_affordance'], local_rank=local_rank)

    criterion_hm = HM_Loss()
    criterion_ce = nn.CrossEntropyLoss()

    model = model.to(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    criterion_hm = criterion_hm.to(local_rank)
    criterion_ce = criterion_ce.to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=dict['lr'], betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.decay_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dict['Epoch'], eta_min=1e-6)

    start_epoch = -1

    best_AUC = 0
    '''
    Training
    '''
    for epoch in range(start_epoch+1, opt.Epoch):
        log_string(f'Epoch:{epoch} strat-------')
        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        log_string(f'lr_rate:{learning_rate}')
        train_sampler.set_epoch(epoch)
        num_batches = len(train_loader)
        loss_sum = 0
        total_point = 0
        model = model.train()

        for i,(img, points, labels, logits_labels, sub_box, obj_box) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                temp_loss = 0

                for point, label, logits_label in zip(points, labels, logits_labels):

                    point, label = point.float(), label.float()
                    label = label.unsqueeze(dim=-1)

                    if(opt.use_gpu):
                        img = img.to(local_rank)
                        point = point.to(local_rank)
                        label = label.to(local_rank)
                        logits_label = logits_label.to(local_rank)
                        sub_box = sub_box.to(local_rank)
                        obj_box = obj_box.to(local_rank)
        
                    _3d, logits, to_KL = model(img, point, sub_box, obj_box)

                    loss_hm = criterion_hm(_3d, label)
                    loss_ce = criterion_ce(logits, logits_label)
                    loss_kl = kl_div(to_KL[0], to_KL[1])
                    temp_loss += loss_hm + opt.loss_cls*loss_ce + opt.loss_kl*loss_kl

                if(opt.local_rank==0):
                    print(f'Epoch:{epoch} | iteration:{i} | loss:{temp_loss.item()}')

                loss_sum += temp_loss.item()
                temp_loss.backward()
                optimizer.step()

        mean_loss = loss_sum / (num_batches*dict['pairing_num'])

        log_string(f'Epoch:{epoch} | mean_loss:{mean_loss}')

        if(opt.storage == True):
            if((epoch+1) % 10==0 and opt.local_rank==0):
                model_path = save_path + '/Epoch_' + str(epoch+1) + '.pt'
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'Epoch': epoch
                }
                torch.save(checkpoint, model_path)
                log_string(f'model saved at {model_path}')
        
        results = torch.zeros((len(val_dataset), 2048, 1))
        targets = torch.zeros((len(val_dataset), 2048, 1))
        '''
        Evalization
        '''
        if((epoch+1)%1 == 0):
            num = 0
            with torch.no_grad():
                log_string(f'EVALUATION strat-------')
                num_batches = len(val_loader)
                total_MAE = 0
                total_point = 0
                model = model.eval()
                for i,(img, point, label,_,_,sub_box, obj_box) in enumerate(val_loader):
                    print(f'iteration: {i} start----')
                    point, label = point.float(), label.float()
                    label = label.unsqueeze(dim=-1)
                    if(opt.use_gpu):
                        img = img.to(local_rank)
                        point = point.to(local_rank)
                        label = label.to(local_rank)
                        sub_box = sub_box.to(local_rank)
                        obj_box = obj_box.to(local_rank)
                    
                    _3d, logits, to_KL = model(img, point, sub_box, obj_box)

                    mae, point_nums = evaluating(_3d, label)
                    total_point += point_nums
                    total_MAE += mae.item()
                    pred_num = _3d.shape[0]

                    results[num : num+pred_num, :, :] = _3d
                    targets[num : num+pred_num, :, :] = label
                    num += pred_num

                mean_mae = total_MAE / total_point
                results = results.detach().numpy()
                targets = targets.detach().numpy()

                SIM_matrix = np.zeros(targets.shape[0])
                for i in range(targets.shape[0]):
                    SIM_matrix[i] = SIM(results[i], targets[i])
                

                sim = np.mean(SIM_matrix)
                AUC = np.zeros((targets.shape[0], targets.shape[2]))
                IOU = np.zeros((targets.shape[0], targets.shape[2]))
                IOU_thres = np.linspace(0, 1, 20)
                targets = targets >= 0.5
                targets = targets.astype(int)
                for i in range(AUC.shape[0]):
                    t_true = targets[i]
                    p_score = results[i]

                    if np.sum(t_true) == 0:
                        AUC[i] = np.nan
                        IOU[i] = np.nan
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

                AUC = np.nanmean(AUC)
                IOU = np.nanmean(IOU)

                log_string(f'AUC:{AUC} | IOU:{IOU} | SIM:{sim} | MAE:{mean_mae}')
                current_AUC = AUC
                if(current_AUC > best_AUC and opt.local_rank==0):
                    best_AUC = current_AUC
                    best_model_path = save_path + '/best.pt'
                    checkpoint = {
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'Epoch': epoch
                    }
                    torch.save(checkpoint, best_model_path)
                    log_string(f'best model saved at {best_model_path}')
        scheduler.step()

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu device id')
    parser.add_argument('--decay_rate', type=float, default=1e-3, help='weight decay [default: 1e-3]')
    parser.add_argument('--use_gpu', type=str, default=True, help='whether or not use gpus')
    parser.add_argument('--save_dir', type=str, default='runs/train/', help='path to save .pt model while training')
    parser.add_argument('--name', type=str, default='IAG', help='training name to classify each training process')
    parser.add_argument('--log_name', type=str, default='train.log', help='the name of current training log')
    parser.add_argument('--loss_cls', type=float, default=0.3, help='cls loss scale')
    parser.add_argument('--loss_kl', type=float, default=0.5, help='kl loss scale')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank devices id')
    parser.add_argument('--gpu_num', type=int, default=4, help='gpu_num')
    opt = parser.parse_args()
    seed_torch(seed=42)
    torch.distributed.init_process_group('nccl', world_size=opt.gpu_num, rank=opt.local_rank)
    dict = read_yaml('config/config_seen.yaml')
    main(opt, dict)
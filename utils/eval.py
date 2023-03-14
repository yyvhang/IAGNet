import numpy as np
import torch

def evaluating(pred, label):

    mae = torch.sum(torch.abs(pred-label), dim=(0,1))
    points_num = pred.shape[0] * pred.shape[1]

    return mae, points_num

def KLD(map1, map2, eps = 1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    kld = np.sum(map2*np.log( map2/(map1+eps) + eps))
    return kld
    
def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

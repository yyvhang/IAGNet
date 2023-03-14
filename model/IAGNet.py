
import numpy
import torch
import pdb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from torchvision.ops import roi_pool, roi_align

class Cross_Attention(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.proj_q = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sk = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sv = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ek = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ev = nn.Linear(self.emb_dim, proj_dim)
        self.scale = self.proj_dim ** (-0.5)

        self.layernorm = nn.LayerNorm(self.emb_dim)
    def forward(self, obj, sub, scene):
        '''
        obj: [B,N_p+HW,C]
        others : [B, HW, C]
        '''
        B, seq_length, C = obj.size()
        query = self.proj_q(obj)                                         #[B, N_q, proj_dim]
        s_key = self.proj_sk(sub)                                        #[B, N_i, proj_dim]
        s_value = self.proj_sv(sub)

        e_key = self.proj_ek(scene)
        e_value = self.proj_ev(scene)

        atten_I1 = torch.bmm(query, s_key.mT)*self.scale                 #[B, N_q, N_i]
        atten_I1 = atten_I1.softmax(dim=-1)
        I_1 = torch.bmm(atten_I1, s_value)

        atten_I2 = torch.bmm(query, e_key.mT)*self.scale                 #[B, N_q, N_i]
        atten_I2 = atten_I2.softmax(dim=-1)
        I_2 = torch.bmm(atten_I2, e_value)

        I_1 = self.layernorm(obj + I_1)                                  #[B, N_q+N_i, emb_dim]
        I_2 = self.layernorm(obj + I_2)
        return I_1, I_2
    
class Inherent_relation(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Inherent_relation, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        

        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, seq_len, head_dim)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)         # (batch_size, num_heads, seq_len, head_dim)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_size ** 0.5)                  # (batch_size, num_heads, seq_len, seq_len)
        
        # Apply softmax activation to scores
        attention_weights = nn.functional.softmax(scores, dim=-1)                                           # (batch_size, num_heads, seq_len, seq_len)

        # Apply attention weights to values
        out = torch.matmul(attention_weights, values)                                                       # (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate and reshape heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)                         # (batch_size, seq_len, embed_dim)
        
        # Apply layer normalization
        out = self.ln(out + x)
        return out


class Joint_Region_Alignment(nn.Module):
    def __init__(self, emb_dim = 512, num_heads = 4):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.emb_dim = emb_dim
        self.div_scale = self.emb_dim ** (-0.5)
        self.num_heads = num_heads

        self.to_common = nn.Sequential(
            nn.Conv1d(self.emb_dim, 2*self.emb_dim, 1, 1),
            nn.BatchNorm1d(2*self.emb_dim),
            nn.ReLU(),
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()         
        )

        self.i_atten = Inherent_relation(self.emb_dim, self.num_heads)
        self.p_atten = Inherent_relation(self.emb_dim, self.num_heads)
        self.joint_atten = Inherent_relation(self.emb_dim, self.num_heads)

    def forward(self, F_i, F_p):
        '''
        i_feature: [B, C, H, W]
        p_feature: [B, C, N_p]
        HW = N_i
        '''

        B,_,N_p = F_p.size()
        F_i = F_i.view(B, self.emb_dim, -1)                                             #[B, C, N_i]

        I = self.to_common(F_i)
        P = self.to_common(F_p)

        phi = torch.bmm(P.permute(0, 2, 1), I)*self.div_scale                           #[B, N_p, N_i]
        phi_p = F.softmax(phi,dim=1)
        phi_i = F.softmax(phi,dim=-1)  
        I_enhance = torch.bmm(P, phi_p)                                                 #[B, C, N_i]
        P_enhance = torch.bmm(I, phi_i.permute(0,2,1))                                  #[B, C, N_p]
        I_ = self.i_atten(I_enhance.mT)                                                 #[B, N_i, C]
        P_ = self.p_atten(P_enhance.mT)                                                 #[B, N_p, C]

        joint_patch = torch.cat((P_, I_), dim=1)                                       
        F_j = self.joint_atten(joint_patch)                                             #[B, N_p+N_i, C]

        return F_j


class Affordance_Revealed_Module(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.cross_atten = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )

    def forward(self, F_j, F_s, F_e):

        '''
        F_j: [B, N_p + N_i, C]
        F_s: [B, H, W, C]
        F_e: [B, H, W, C]
        '''

        B,_,C = F_j.size()

        F_s = F_s.view(B, C, -1)                                        #[B, N_i, C]
        F_e = F_e.view(B, C, -1)                                        #[B, N_i, C]
        Theta_1, Theta_2 = self.cross_atten(F_j, F_s.mT, F_e.mT)        #[B, C, N_p + N_i]

        joint_context = torch.cat((Theta_1.mT, Theta_2.mT), dim=1)      #[B, 2C, N_p + N_i]
        affordance = self.fusion(joint_context)                         #[B, C, N_p + N_i]
        affordance = affordance.permute(0, 2, 1)                        #[B, N_p + N_i, C]

        return affordance

class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()

        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]

class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        self.model = models.resnet18(weights=None)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        B, _, _, _ = img.size()
        out = self.model.conv1(img)
        out = self.model.relu(self.model.bn1(out))

        out = self.model.maxpool(out) 
        out = self.model.layer1(out)   

        down_1 = self.model.layer2(out)         

        down_2 = self.model.layer3(down_1)       

        down_3 = self.model.layer4(down_2)

        return down_3

class Decoder(nn.Module):
    def __init__(self, additional_channel, emb_dim, N_p, N_raw, num_affordance):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()
        
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N = N_raw
        self.num_affordance = num_affordance
        #upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])  
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512]) 
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            SwapAxes(),
            nn.BatchNorm1d(self.emb_dim // 8),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(self.emb_dim // 8, 1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(2*self.emb_dim, self.emb_dim // 2),
            nn.BatchNorm1d(self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, self.num_affordance),
            nn.BatchNorm1d(self.num_affordance)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, F_j, affordance, encoder_p):

        '''
        obj --->        [F_j]
        affordance ---> [B, N_p + N_i, C]
        encoder_p  ---> [Hierarchy feature]
        '''
        B,_,_ = F_j.size()
        p_0, p_1, p_2, p_3 = encoder_p
        P_align, I_align = torch.split(F_j, split_size_or_sections=self.N_p, dim=1)     #[B, N_p, C] --- [B, N_i, C]
        F_pa, F_ia = torch.split(affordance, split_size_or_sections = self.N_p, dim=1)  #[B, N_p, C] --- [B, N_i, C]

        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], P_align.mT)                        #[B, emb_dim, npoint_sa2]
        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)                         #[B, emb_dim, npoint_sa1]                        
        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample)  #[B, emb_dim, N]

        F_pa_pool = self.pool(F_pa.mT)                                                  #[B, emb_dim, 1]
        F_ia_pool = self.pool(F_ia.mT)                                                  #[B, emb_dim, 1]
        logits = torch.cat((F_pa_pool, F_ia_pool), dim=1)                               #[B, 2*emb_dim, 1]
        logits = self.cls_head(logits.view(B,-1))

        _3daffordance = up_sample * F_pa_pool.expand(-1,-1,self.N)                      #[B, emb_dim, 2048]
        _3daffordance = self.out_head(_3daffordance.mT)                                    #[B, 2048, 1]
        _3daffordance = self.sigmoid(_3daffordance)

        return _3daffordance, logits, [F_ia.mT.contiguous(), I_align.mT.contiguous()]


class IAG(nn.Module):
    def __init__(self, img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=17):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.num_affordance = num_affordance
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.img_encoder = Img_Encoder()
        if pre_train:
            pretrain_dict = torch.load(img_model_path)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)

        self.JRA = Joint_Region_Alignment(self.emb_dim, self.num_heads)
        self.ARM = Affordance_Revealed_Module(self.emb_dim, self.proj_dim)

        self.decoder = Decoder(self.additional_channel, self.emb_dim, self.N_p, self.N_raw, self.num_affordance)
        
    def forward(self, img, xyz, sub_box, obj_box):

        '''
        img: [B, 3, H, W]
        xyz: [B, 3, 2048]
        sub_box: bounding box of the interactive subject
        obj_box: bounding box of the interactive object
        '''

        B, C, N = xyz.size()
        if(self.local_rank != None):
            device = torch.device('cuda', self.local_rank)
        else:
            device = torch.device('cuda:0')

        F_I = self.img_encoder(img)
        ROI_box = self.get_roi_box(B).to(device)

        F_i, F_s, F_e = self.get_mask_feature(img, F_I, sub_box, obj_box, device)     #[B, 512, 7, 7]  
        F_e = roi_align(F_e, ROI_box, output_size=(4,4))

        F_p_wise = self.point_encoder(xyz)
        F_j = self.JRA(F_i, F_p_wise[-1][1])
        affordance = self.ARM(F_j, F_s, F_e)

        _3daffordance, logits, to_KL = self.decoder(F_j, affordance, F_p_wise)

        return _3daffordance, logits, to_KL

    def get_mask_feature(self, raw_img, img_feature, sub_box, obj_box, device):
        raw_size = raw_img.size(2)
        current_size = img_feature.size(2)
        B = img_feature.size(0)
        scale_factor = current_size / raw_size

        sub_box[:, :] = sub_box[:, :] * scale_factor
        obj_box[:, :] = obj_box[:, :] * scale_factor

        obj_mask = torch.zeros_like(img_feature)
        obj_roi_box = []
        for i in range(B):
            obj_mask[i,:, int(obj_box[i][1]+0.5):int(obj_box[i][3]+0.5), int(obj_box[i][0]+0.5):int(obj_box[i][2]+0.5)] = 1
            roi_obj = [obj_box[i][0], obj_box[i][1], obj_box[i][2]+0.5, obj_box[i][3]]
            roi_obj.insert(0, i)
            obj_roi_box.append(roi_obj)
        obj_roi_box = torch.tensor(obj_roi_box).float().to(device)

        sub_roi_box = []

        Scene_mask = obj_mask.clone()
        for i in range(B):
            Scene_mask[i,:, int(sub_box[i][1]+0.5):int(sub_box[i][3]+0.5), int(sub_box[i][0]+0.5):int(sub_box[i][2]+0.5)] = 1
            roi_sub = [sub_box[i][0], sub_box[i][1], sub_box[i][2], sub_box[i][3]]
            roi_sub.insert(0,i)
            sub_roi_box.append(roi_sub)
        Scene_mask = torch.abs(Scene_mask - 1)
        Scene_mask_feature = img_feature * Scene_mask
        sub_roi_box = torch.tensor(sub_roi_box).float().to(device)
        obj_feature = roi_align(img_feature, obj_roi_box, output_size=(4,4), sampling_ratio=4)
        sub_feature = roi_align(img_feature, sub_roi_box, output_size=(4,4), sampling_ratio=4) 
        return obj_feature, sub_feature, Scene_mask_feature

    def get_roi_box(self, batch_size):
        batch_box = []
        roi_box = [0., 0., 6., 6.]
        for i in range(batch_size):
            roi_box.insert(0, i)
            batch_box.append(roi_box)
            roi_box = roi_box[1:]

        batch_box = torch.tensor(batch_box).float()

        return batch_box

def get_IAGNet(img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=17):
    
    model = IAG(img_model_path, pre_train, normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, N_raw, num_affordance)
    return model
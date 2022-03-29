import math
from sklearn import preprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from torch import einsum
from GCN_Unsupervised.rotation import *
from itertools import permutations
from einops import rearrange, repeat
import GCN_Unsupervised.GCNEncoder.ctrgcnEncoder as ctrgcnEncoder
import GCN_Unsupervised.GCNEncoder.shiftGCNEncoder as shiftGCNEncoder
import GCN_Unsupervised.GCNEncoder.AGCNEncoder as AGCNEncoder
import copy

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.l3 = nn.Linear(128,3)

    def forward(self,x):
        N, M,C,T,V = x.size()
        x = x.reshape(N*M,C,T,V)
        x = x.view(N,M,128,T,V).permute(0,1,3,4,2)
        x = self.l3(x)
        x = x.permute(0,4,2,3,1)
        return x


class AutoEncoder(nn.Module):
    """
    This is the encode unit and decode unit in paper
    """
    def __init__(self,T):
        super(AutoEncoder, self).__init__()
        self.encoder_l1 = nn.Sequential(
            nn.Linear(1600, 128),
            nn.Tanh()
        )
        self.encoder_l2 = nn.Sequential(
            nn.Linear(T, 30),
            nn.Tanh()
        )
        self.decoder_l1 = nn.Sequential(
            nn.Linear(30, T),
            nn.Tanh()
        )
        self.decoder_l2 = nn.Sequential(
            nn.Linear(128, 1600)
        )

    def forward(self, x):
        N, M,C,T,V = x.size()
        x = x.permute(0,1,3,2,4).reshape(N,M,T,C*V)
        x = self.encoder_l1(x)
        x = x.permute(0,1,3,2)
        x = self.encoder_l2(x)
        hidden_feature = x.reshape(N,-1)
        x = self.decoder_l1(x)
        x = x.permute(0,1,3,2)
        x = self.decoder_l2(x)
        x = x.reshape(N,M,T,C,V).permute(0,1,3,2,4)
        return x, hidden_feature


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        N, T, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'n t (h d) -> n h t d', h=h), qkv)

        dots = einsum('n h i d, n h j d -> n h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('n h i j, n h j d -> n h i d', attn, v)
        out = rearrange(out, 'n h t d -> n t (h d)')
        out = self.to_out(out)
        return out


class ATULayer(nn.Module):
    """
    This is the TAU in paper
    """
    def __init__(self,T):
        super(ATULayer,self).__init__()
        self.attn = Residual(PreNorm(1600, Attention(1600, heads=8,dim_head=128, dropout=0.)))
        self.linear = nn.Linear(T,T)
        self.Tanh =  nn.Tanh()

    def forward(self,x):
        N,M,C,V,T = x.shape
        x = x.permute(0,1,4,2,3).reshape(N*M,T,C*V)
        x = self.attn(x)
        x = x.reshape(N,M,T,C,V).permute(0,1,3,4,2)
        x = self.linear(x) # N,M,C,V,T
        x = self.Tanh(x)
        return x


class ATNet(nn.Module):
    def __init__(self,layers,T):
        super(ATNet,self).__init__()
        self.layer = layers
        ATUlayer = ATULayer(T)
        self.ATUNet = nn.ModuleList([copy.deepcopy(ATUlayer) for _ in range(self.layer)])

    def forward(self,x):
        N,M,C,T,V = x.shape
        x = x.permute(0,1,2,4,3)
        for ATU in self.ATUNet:
            x = ATU(x)
        x = x.permute(0,1,2,4,3)
        return x

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,self_supervised_mask=1,if_rotation=False,seg_num=1,if_vibrate=False,prediction_mask=0,GCNEncoder="AGCN",ATU_layer=2,T=100,predict_seg=1):
        super(Model, self).__init__()
        self.lefthand = shiftGCNEncoder.ShiftGCNEncoder(num_class, 6, num_person, graph, graph_args,in_channels)
        self.righthand = shiftGCNEncoder.ShiftGCNEncoder(num_class, 6, num_person, graph, graph_args,in_channels)
        self.trunk = shiftGCNEncoder.ShiftGCNEncoder(num_class, 5, num_person, graph, graph_args,in_channels)
        self.leftleg = shiftGCNEncoder.ShiftGCNEncoder(num_class, 4, num_person, graph, graph_args,in_channels)
        self.rightleg = shiftGCNEncoder.ShiftGCNEncoder(num_class, 4, num_person, graph, graph_args,in_channels)
        self.LEFT_HAND = np.asarray([5,6,7,8,22,23])
        self.RIGHT_HAND = np.asarray([9,10,11,12,24,25])
        self.TRUNK = np.asarray([4,3,21,2,1])
        self.LEFT_LEG = np.asarray([13,14,15,16])
        self.RIGHT_LEG = np.asarray([17,18,19,20])


        if GCNEncoder == "CTRGCN":
            self.Encoder = ctrgcnEncoder.CTRGCNEncoder(num_class, num_point, num_person, graph, graph_args,in_channels)
        elif GCNEncoder == "AGCN":
            self.Encoder = AGCNEncoder.AGCNEncoder(num_class, num_point, num_person, graph, graph_args,in_channels)
        elif GCNEncoder == "shiftGCN":
            self.Encoder = shiftGCNEncoder.ShiftGCNEncoder(num_class, num_point, num_person, graph, graph_args,64)
        self.autoEncoder = AutoEncoder(T)
        self.part_autoEncoder = AutoEncoder(T)
        self.Decoder = Decoder()
        self.selfsupervised_mask = self_supervised_mask
        self.if_rotation = if_rotation
        self.if_vibrate = if_vibrate
        self.prediction_mask = prediction_mask
        self.AttentionTemporalNet = ATNet(ATU_layer,T)
        self.Part_AttentionTemporalNet = ATNet(ATU_layer,T)
        self.predict_seg = predict_seg
        self.JointDirectionClassifier = JointDirectionClassifier()

    def forward(self,x,is_test=False):
        N, C, T, V, M = x.size()
        prediction_length = T*self.prediction_mask
        if is_test == False:
            # 对样本随机掩码
            predict_seg_length = T//self.predict_seg
            for i in range(1,self.predict_seg+1):
                x[:,:,int(i*predict_seg_length-prediction_length):int(i*predict_seg_length),:,:] = 0


        GCN_feature = torch.empty(N,M,64,T,V).cuda().to(torch.float32) # N,M,C,T,V
        GCN_feature[:,:,:,:,self.LEFT_HAND-1] = self.lefthand(x[:,:,:,self.LEFT_HAND-1,:])
        GCN_feature[:,:,:,:,self.RIGHT_HAND-1] = self.righthand(x[:,:,:,self.RIGHT_HAND-1,:])
        GCN_feature[:,:,:,:,self.TRUNK-1] = self.trunk(x[:,:,:,self.TRUNK-1,:])
        GCN_feature[:,:,:,:,self.LEFT_LEG-1] = self.leftleg(x[:,:,:,self.LEFT_LEG-1,:])
        GCN_feature[:,:,:,:,self.RIGHT_LEG-1] = self.rightleg(x[:,:,:,self.RIGHT_LEG-1,:])
        part_feature = GCN_feature #N,M,C,T,V
        GCN_feature = GCN_feature.permute(0,2,3,4,1) #N,C,T,V,M
        GCN_feature = self.Encoder(GCN_feature)
        #NMCTV
        GCN_feature = self.AttentionTemporalNet(GCN_feature)
        part_feature = self.Part_AttentionTemporalNet(part_feature)
        part_autoencoder_output_feature,part_hidden_featurepart_feature = self.part_autoEncoder(part_feature) 
        autoencoder_output_feature,hidden_feature = self.autoEncoder(GCN_feature)
        autoencoder_output_feature = torch.cat((autoencoder_output_feature,part_autoencoder_output_feature),2)
        output = self.Decoder(autoencoder_output_feature)
        
        hidden_feature = torch.cat((hidden_feature,part_hidden_featurepart_feature),1)
        return output,hidden_feature,GCN_feature,autoencoder_output_feature,None


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        num_class = 60
        self.l1 = nn.Sequential(
            nn.Linear(15360, 2048)
        ) 
        self.l2 = nn.Sequential(
            nn.Linear(2048, 512)
        ) 
        self.l3 = nn.Sequential(
            nn.Linear(512, 60)
        ) 

    def forward(self,x):                
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x

class CustomizeL2Loss(nn.Module):
    def __init__(self):
        super(CustomizeL2Loss,self).__init__()
        # self.ratio = torch.tensor([1 for i in range(25)])
        self.ratio = 25 * torch.tensor([0.07716928,0.03395055,0.07363877,0.08779999,0.09124018,0.15031101,0.21393532,0.2563119,0.09196077,0.15753494,0.22839014,0.27403742,
        0.0905839 ,0.14892179 ,0.17746575, 0.25223324, 0.09057538, 0.15295361,0.17971317 ,0.25437343 ,0.06341638, 0.30254194 ,0.32573557 ,0.3223823, 0.34228417])
        
    def forward(self,x,y):
        #NCTVM
        N,C,T,V,M = x.shape
        motion = x[:,:,1:,...]-x[:,:,:T-1,...]
        mean_move = torch.empty(N,25)
        for j in range(N):
            for i in range(25):
                mean_move[j][i] = torch.mean(torch.abs(motion[j,:,:,i,:]))
            mean_move[j] = 25*mean_move[j].clone()/torch.sum(mean_move[j].clone())
        ratio = mean_move.cuda()
        return torch.mean(einsum("nctvm,nv->nctvm",torch.pow((x-y),2),ratio))
     
class BoneLengthLoss(nn.Module):
    def __init__(self):
        super(BoneLengthLoss,self).__init__()
        self.NTU_POSE_EDGES = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),(14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12))

    def forward(self,x,y):
        N,C,T,V,M = x.shape
        bone1 = torch.zeros((N,C,T,V,M),dtype=torch.float32)
        bone2 = torch.zeros((N,C,T,V,M),dtype=torch.float32)
        for v1,v2 in self.NTU_POSE_EDGES:
            bone1[:,:,:,v1-1,:] = x[:,:,:,v1-1,:]- x[:,:,:,v2-1,:]
            bone2[:,:,:,v1-1,:] = y[:,:,:,v1-1,:]- y[:,:,:,v2-1,:]
        return torch.mean(torch.abs(bone1-bone2))

class MotionLoss(nn.Module):
    def __init__(self):
        super(MotionLoss,self).__init__()

    def forward(self,x,y):
        N,C,T,V,M = x.shape
        #NCTVM
        N,C,T,V,M = x.shape
        motion = x[:,:,1:,...]-x[:,:,:T-1,...]
        mean_move = [[0 for i in range(25)]]
        for i in range(25):
            mean_move[0][i] = torch.mean(torch.abs(motion[:,:,:,i,:]))
        self.ratio = 25*torch.tensor(preprocessing.normalize(np.asarray(mean_move))[0]).cuda()

        motion1 = x[:,:,1:,:,:] - x[:,:,:T-1,:,:]
        motion2 = y[:,:,1:,:,:] - y[:,:,:T-1,:,:]
        return torch.mean(einsum("nctvm,v->nctvm",torch.pow((motion1-motion2),2),self.ratio))


class JointDirectionPredictionLoss(nn.Module):
    def __init__(self):
        super(JointDirectionPredictionLoss,self).__init__()

    def forward(self,y,yhat):
        #y: N,V,27
        # yhat = yhat.cuda()
        # y = y.cuda()
        yhat = nn.functional.softmax(yhat,2) #先将其归一化，再求交叉熵
        return torch.mean(einsum("nvc,nvc->nv",y,yhat))
        
   
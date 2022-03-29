import numpy as np
import math
import random
import torch.nn as nn
import torch

x = np.zeros((5,10))
print(x)
y = np.random.rand(10)
print(y)
x = x+y
print(x)
# N,C,T,V,M = 3,3,100,2,1
# x = np.random.rand(N,C,T,V,M)

# # print(x)
# index = np.random.choice(np.arange(0,T,1),size=(int(T*0.5)),replace=False)
# index.sort(0)
# print(index)
# print(index.shape)
# print(x.shape)
# # x = x[index]
# x[:,:,index,...] = 0
# print(x)
# # x = x[:,:,index,...]
# # print(x)
# print(x.shape)
# print(x)
# mask = np.random.binomial(1, p=0.5,size=(1,T,1,1))
# x = x * mask
# print()
# print(x)
# N = 3
# V=1
# T = 30
# C=3
# M=1
# colorization = np.zeros((N,C,T,V,M))
# data = np.random.rand(N,C,T,V)
# for t in range(T):
#     if t<=T/2:
#         colorization[:,0,t,:] = -2*(t/T)+1
#         colorization[:,1,t,:] = 2*(t/T)
#     else:
#         colorization[:,1,t,:] = -2*(t/T)+2
#         colorization[:,2,t,:] = 2*(t/T)-1
# print(colorization)
# def f(a):
#     a[0] = 0

# N,V = 5,10
# input = torch.randn(N, V,2)
# a = input
# input = 0
# print(a)
# f(input)
# print(input)
# N,V = 5,10
# input = torch.randn(N, V,2)
# # output = nn.Softmax((input)
# output = nn.functional.softmax(input,2)
# print(output)
# def softmax(logits):
#     return np.exp(logits)/np.sum(np.exp(logits))
# a = np.asarray([1,0,2])
# print(softmax(a))
# def joint_label(joint_motion):
#     label = 0
#     for i in joint_motion:
#         label = label*3
#         if i == 0.0:
#             label +=1
#         elif i==1.0:
#             label +=2
#     return label


# N,T,C,V =2,5,3,5 
# x = np.random.rand(N,C,T,V)
# print(x)
# label = np.zeros((N,V,27))
# for i in range(N):
#     t = random.randint(1,4)
#     print(t)
#     motion = np.sign(x[i,:,t,:]-x[i,:,t-1,:])
#     for j,joint_motion in enumerate(motion.T):
#         L = joint_label(joint_motion)
#         label[i,j,L] = 1
# print(label)
    
            


# LEFT_HAND = np.asarray([5,6,7,8,22,23])
# RIGHT_HAND = np.asarray([9,10,11,12,24,25])
# TRUNK = np.asarray([4,3,21,2,1])
# LEFTLEG = np.asarray([13,14,15,16])
# RIGHTLEG = np.asarray([17,18,19,20])
# N,V =5,25 
# x = np.random.rand(N,V)
# x[:,LEFT_HAND-1] = 1
# x[:,RIGHT_HAND-1] = 2
# print(x)


# '''
# 骨骼归一化
# '''

# _NTU_POSE_EDGES = [
#     (1, 2), (2, 21), (21, 3), (3, 4),
#     (21, 9), (9, 10), (10, 11), (11, 12), (12, 24), (12, 25),
#     (21, 5), (5, 6), (6, 7), (7, 8), (8, 22), (8, 23),
#     (1, 17), (17, 18), (18, 19), (19, 20),
#     (1, 13), (13, 14), (14, 15), (15, 16)
# ]
# # M,T,V,C = skeleton.shape

# def read_data(src_path):
#     '''
#     :param src_path: 骨架文件路径
#     :return: 骨架坐标点的list
#     '''
#     all_point_loc = []
#     with open(src_path,'r') as f:
#         frame_num = int(f.readline())
#         for i in range(frame_num):
#             body_num = int(f.readline())
#             body_loc = []
#             for j in range(body_num):
#                 f.readline()
#                 f.readline()
#                 for k in range(25):
#                     point_info = f.readline().split(' ')
#                     if(j>=1): continue
#                     point_x_y_z = []
#                     for temp in point_info[:3]:
#                         temp = temp.strip()
#                         temp = float(temp)
#                         point_x_y_z.append(temp)
#                     body_loc.append(point_x_y_z.copy())
#             all_point_loc.append(body_loc.copy())
#     # T V C
#     return all_point_loc

# def normalize(skeleton,nor = 1):
#     # 求骨骼长度和
#     bone_length = 0
#     for j,k in _NTU_POSE_EDGES:
#         bone_length += np.sum(skeleton[0,0,j-1] - skeleton[0,0,k-1])
#     skeleton = skeleton/bone_length * nor
#     print(bone_length)
#     return skeleton

# src1 = '/root/dataset/nturgb+d_skeletons/S005C002P004R001A002.skeleton'
# data1 = np.zeros((2,300,25,3), dtype=np.float32)
# t = np.asarray(read_data(src1))
# data1[0,:t.shape[0],...] = t
# normalize(data1)

# # def f(a):
# #     b = np.transpose(a,(1,0))
# #     b[0][0] = 100
# #     a = np.transpose(b,(1,0))
# #     return a

# # a = np.asarray([[0,1,2],[3,4,5]])
# # x = f(a.copy())

# # print(a)
# # print(x)

# # def preSampleLen(data,max_frame):
# #     """
# #     C,T,V,M
# #     仅保留骨架中不为零的骨架
# #     设置一个最大长度M，此外如果骨架序列的长度比M大，那么进行降采样，如果比M小，那么采用复制填充法
# #     remove the zero frame and then make the sample into target length
# #     :param data:
# #     :param target_frame:
# #     :return:
# #     """
# #     C,T,V,M = data.shape
# #     data = np.transpose(data,[3,1,2,0]) # M,T,V,C
# #     for i_p, person in enumerate(data):
# #             if person.sum() == 0:
# #                 continue
# #             # 保留不为零的帧
# #             index = (person.sum(-1).sum(-1) != 0)
# #             nonzero_frame = person[index].copy() # T,V,C
    
# #             new_data = np.zeros((M,max_frame,V,C))
# #             if len(nonzero_frame) > max_frame:
# #                 # 降采样
# #                 diff = math.floor(len(nonzero_frame) / max_frame)
# #                 idx = 0
# #                 for i in range(0, len(nonzero_frame), diff):
# #                     new_data[i_p,idx,:,:] = nonzero_frame[i]
# #                     idx += 1
# #                     if idx >= max_frame:
# #                         break
# #             else:
# #                 #长度不足的序列采用复制填充法
# #                 new_data[i_p,:len(nonzero_frame),:,:] = nonzero_frame
# #                 num = int(np.ceil(max_frame / len(nonzero_frame)))
# #                 new_data[i_p] = np.concatenate([nonzero_frame for _ in range(num)])[:max_frame]
# #     new_data = np.transpose(new_data,[3,1,2,0]) # C,T,V,M
# #     return new_data


# # # M,T,V,C
# # # a = np.asarray([])

# # a = np.asarray([[[1,0,3]],[[0,0,0]]])
# # print(a.shape)
# # b = a.sum(-1)!=0
# # print(b.shape)


# # def preSampleLen(data,target_frame):
# #     """
# #     C,T,V,M
# #     仅保留骨架中不为零的骨架
# #     设置一个最大长度M，此外如果骨架序列的长度比M大，那么进行降采样，如果比M小，那么采用复制填充法
# #     remove the zero frame and then make the sample into target length
# #     """
# #     C,T,V,M = data.shape
# #     data = np.transpose(data,[3,1,2,0]) # M,T,V,C
# #     new_data = np.zeros((M,target_frame,V,C))
# #     for i_p, person in enumerate(data):
# #             if person.sum() == 0:
# #                 continue
# #             # 保留不为零的帧
# #             index = (person.sum(-1).sum(-1) != 0)
# #             nonzero_frame = person[index].copy() # T,V,C
# #             if len(nonzero_frame) > target_frame:
# #                 # 降采样
# #                 diff = math.floor(len(nonzero_frame) / target_frame)
# #                 idx = 0
# #                 for i in range(0, len(nonzero_frame), diff):
# #                     new_data[i_p,idx,:,:] = nonzero_frame[i]
# #                     idx += 1
# #                     if idx >= target_frame:
# #                         break
# #             else:
# #                 #长度不足的序列采用复制填充法
# #                 num = int(np.ceil(target_frame / len(nonzero_frame)))
# #                 new_data[i_p] = np.concatenate([nonzero_frame for _ in range(num)])[:target_frame]
                
# #     new_data = np.transpose(new_data,[3,1,2,0]) # C,T,V,M
# #     return new_data

# # data = np.random.rand(2,7,2,2)
# # data[:,2,:,:] = 0
# # print(data)

# # print("*************************")
# # data = preSampleLen(data,5)
# # print(data)
# # C,T,V,M = data.shape



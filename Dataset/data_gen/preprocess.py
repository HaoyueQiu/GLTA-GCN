import sys

sys.path.extend(['../'])
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np
import math

_NTU_POSE_EDGES = [
    (1, 2), (2, 21), (21, 3), (3, 4),
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 24), (12, 25),
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 22), (8, 23),
    (1, 17), (17, 18), (18, 19), (19, 20),
    (1, 13), (13, 14), (14, 15), (15, 16)
]


def unit_vector(vector):
    """ 
        返回单位向量
        Returns the unit vector of the vector.  
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
        返回v1和v2之间需要旋转的角度
        Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def preSampleLen(data,target_frame):
    """
    C,T,V,M
    仅保留骨架中不为零的骨架
    设置一个最大长度M，此外如果骨架序列的长度比M大，那么进行降采样，如果比M小，那么采用复制填充法
    remove the zero frame and then make the sample into target length
    """
    C,T,V,M = data.shape
    data = np.transpose(data,[3,1,2,0]) # M,T,V,C
    new_data = np.zeros((M,target_frame,V,C))
    for i_p, person in enumerate(data):
            if person.sum() == 0:
                continue
            # 保留不为零的帧
            index = (person.sum(-1).sum(-1) != 0)
            nonzero_frame = person[index].copy() # T,V,C
            if len(nonzero_frame) > target_frame:
                # 降采样
                diff = math.floor(len(nonzero_frame) / target_frame)
                idx = 0
                for i in range(0, len(nonzero_frame), diff):
                    new_data[i_p,idx,:,:] = nonzero_frame[i]
                    idx += 1
                    if idx >= target_frame:
                        break
            else:
                #长度不足的序列采用复制填充法
                num = int(np.ceil(target_frame / len(nonzero_frame)))
                new_data[i_p] = np.concatenate([nonzero_frame for _ in range(num)])[:target_frame]
                
    new_data = np.transpose(new_data,[3,1,2,0]) # C,T,V,M
    return new_data

def relative_coordinate(skeleton):
    M,T,V,C = skeleton.shape
    main_body_center = skeleton[0][:, 1:2, :].copy()
    mask = (skeleton.sum(-1) != 0).reshape(M, T, V, 1)
    skeleton = (skeleton - main_body_center) * mask
    return skeleton

def paralle(skeleton,vec1,vec2):
    """
    骨架旋转：将vec1旋转至vec2
    求出两个向量的法向量、夹角-->旋转矩阵，对该骨架进行旋转
    """
    # np.cross: 叉积，所得为两个向量所在平面的法向量，旋转轴
    if np.abs(vec1).sum() < 1e-6 or np.abs(vec2).sum() < 1e-6:
        return skeleton
    axis = unit_vector(np.cross(vec1, vec2))
    angle = angle_between(vec1,vec2)
    # 旋转向量 = 旋转幅度 * 旋转轴(单位向量)
    rotation_matrix = Rotation.from_rotvec(angle * axis).as_matrix() 
    skeleton = np.dot(skeleton,rotation_matrix.T)
    return skeleton

def paralle_z(skeleton):
    """
    对齐z轴,第一帧中的臀部关节点和脊柱关键点
    """
    bottom_top = skeleton[0, 0, 1]-skeleton[0, 0, 0]
    return paralle(skeleton,bottom_top,[0,0,1])

def paralle_x(skeleton):
    """
    对齐x轴,第一帧中的左右肩膀关键点
    """
    shoulder = skeleton[0, 0, 8]-skeleton[0, 0, 4]
    return paralle(skeleton,shoulder,[1,0,0])

def normalize(skeleton,length):
    # 求骨骼长度和
    bone_length = 0
    for j,k in _NTU_POSE_EDGES:
        bone_length += np.linalg.norm(skeleton[0,0,j-1] - skeleton[0,0,k-1])
    skeleton = skeleton/bone_length * length
    print(bone_length)
    return skeleton

def pre_normalization(data):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('sub center & view-invariant transformation')
    for i_s, skeleton in enumerate(tqdm(s)):
        s[i_s] = relative_coordinate(skeleton)
        s[i_s] = paralle_z(skeleton)
        s[i_s] = paralle_x(skeleton)
        s[i_s] = normalize(skeleton,10)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data



if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
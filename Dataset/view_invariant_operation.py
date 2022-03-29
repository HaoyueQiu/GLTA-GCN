# from data_gen.rotation import *
from scipy.spatial.transform import Rotation
import numpy as np
import os

"""
X-View 不同视点下的骨架按照目前骨架对齐方式对齐后的差别有多大？  
实验：统计不同骨架对齐后的差距
"""

T = 300
V = 25

def read_data(src_path):
    '''
    :param src_path: 骨架文件路径
    :return: 骨架坐标点的list
    '''
    all_point_loc = []
    with open(src_path,'r') as f:
        frame_num = int(f.readline())
        for i in range(frame_num):
            body_num = int(f.readline())
            body_loc = []
            for j in range(body_num):
                f.readline()
                f.readline()
                for k in range(25):
                    point_info = f.readline().split(' ')
                    if(j>=1): continue
                    point_x_y_z = []
                    for temp in point_info[:3]:
                        temp = temp.strip()
                        temp = float(temp)
                        point_x_y_z.append(temp)
                    body_loc.append(point_x_y_z.copy())
            all_point_loc.append(body_loc.copy())
    # T V C
    return all_point_loc

def relative_coordinate(skeleton):
    main_body_center = skeleton[:, 1:2, :].copy()
    mask = (skeleton.sum(-1) != 0).reshape(T, V, 1)
    skeleton = (skeleton - main_body_center) * mask
    return skeleton

def unit_vector(v):
    return v/np.linalg.norm(v)

def angle_between(v1,v2):

    # 求向量夹角公式 θ = arccos(v1*v2/(||v1||*||v2||))
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) 

def paralle(skeleton,vec1,vec2):
    """
    骨架旋转：将vec1旋转至vec2
    求出两个向量的法向量、夹角-->旋转矩阵，对该骨架进行旋转
    """
    if np.abs(vec1).sum() < 1e-6 or np.abs(vec2).sum() < 1e-6:
        return skeleton
    # np.cross: 叉积，所得为两个向量所在平面的法向量，旋转轴
    axis = unit_vector(np.cross(vec1, vec2))
    angle = angle_between(vec1,vec2)
    # 旋转向量 = 旋转幅度 * 旋转轴(单位向量)
    rotation_matrix = Rotation.from_rotvec(angle * axis).as_matrix() 
    # dot(a, b)[i,j,k] = sum(a[i,j,:] * b[k,:]),it is a sum product over the last axis of a and the second-to-last axis of b:
    skeleton = np.dot(skeleton,rotation_matrix.T)
    return skeleton

def paralle_z(skeleton):
    """
    对齐z轴,第一帧中的臀部关节点和脊柱关键点
    """
    bottom_top = skeleton[0, 1]-skeleton[0, 0]
    return paralle(skeleton,bottom_top,[0,0,1])

def paralle_x(skeleton):
    """
    对齐x轴,第一帧中的左右肩膀关键点
    """
    shoulder = skeleton[0, 4]-skeleton[0, 8]
    return paralle(skeleton,shoulder,[1,0,0])

def difference(data_src1,data_src2):
    data1 = np.zeros((T,V,3), dtype=np.float32)
    data2 = np.zeros((T,V,3), dtype=np.float32)

    try:
        t = np.asarray(read_data(data_src1))
        data1[:t.shape[0],...] = t
    except:
        print(data_src1)
        return 0,0,0
    try:
        t = np.asarray(read_data(data_src2))
        data2[:t.shape[0],...] = t
    except:
        print(data_src2)
        return 0,0,0


    diff1 = np.sum(np.abs(data1-data2))
    data1 = relative_coordinate(data1)
    data2 = relative_coordinate(data2)
    diff2 = np.sum(np.abs(data1-data2))
    data1 = paralle_x(paralle_z(data1))
    data2= paralle_x(paralle_z(data2))
    diff3 = np.sum(np.abs(data1-data2))
    return diff1,diff2,diff3



# src1 = '/root/dataset/nturgb+d_skeletons/S005C002P004R001A001.skeleton'
# src2 = '/root/dataset/nturgb+d_skeletons/S007C002P017R002A006.skeleton'

# diff1,diff2,diff3 = difference(src1,src2)
# print(diff1,diff2,diff3)
def sta():
    directory = "/root/dataset/nturgb+d_skeletons/"
    sum1 = sum2 = sum3 = 0
    """统计骨架变化前后的关键点坐标差"""
    with open("/root/Transformer_CV/Dataset/missing_skeleton.txt", 'r') as f:
        ignored_samples = [line.strip() + '.skeleton' for line in f.readlines()]

    for filename in os.listdir(directory):
        # print(filename)
        if filename in ignored_samples:
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        if action_class >2: 
            continue
        src1 = directory + filename
        if filename[7] == '1':
            filename = filename[:7]+'2'+filename[8:]
        elif filename[7] == '2':
            filename = filename[:7]+'3'+filename[8:]
        elif filename[7] == '3':
            filename = filename[:7]+'1'+filename[8:]
        if filename in ignored_samples:
            continue
        src2 = directory + filename
        if not os.path.exists(src2):
            continue
        # print(src2)
        diff1,diff2,diff3 = difference(src1,src2)
        sum1 += diff1
        sum2 += diff2
        sum3 += diff3
    print(sum1,sum2,sum3)

sta()

"""
现象记录:
1. 不同视角下的同一骨架序列样本，帧数可能并不一致，比如 S007C002P017R002A006 和  S007C001P017R002A006 分别有89帧和90帧。 三台设备无法同时开启，帧之间本身存在一点点错位。
2. 取相对坐标后两个骨架之间的差距就大大减小了 
3. 旋转前后骨架序列的差距变化不大，推断是由于深度摄像机中对深度这一维度的估量仍然存在一定的不足,导致整个旋转出来的效果就有所不足。
"""
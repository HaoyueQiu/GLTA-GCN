import numpy as np
from sklearn import preprocessing
pairs = [(0, 1), (1, 2),  # 头-肩中心-臀中心
         (1, 3), (3, 4), (4, 5),  # 肩中心-左肩-左肘-左手
         (1, 6), (6, 7), (7, 8),  # 肩中心-右肩-右肘-右手
         (2, 9), (9, 10), (10, 11),  # 臀中心-左臀-左膝-左脚
         (2, 12), (12, 13), (13, 14)]  # 臀中心-右臀-右膝-右脚

data_path = "/root/data/UWA3DTransformedPreprocess/joint_1.npy"
data = np.load(data_path)
N, C, T, V, M = data.shape



motion = data[:,:,1:,...]-data[:,:,:T-1,...]
mean_move = [[0 for i in range(15)]]
for i in range(15):
    mean_move[0][i] = np.mean(np.abs(motion[:,:,:,i,:]))
print(preprocessing.normalize(np.asarray(mean_move)))

# bone = np.zeros((N,C,T,V,M))
# for v1, v2 in pairs:
#     bone[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]


# mean_std = [[0 for i in range(2)] for i in range(25)]
# for i in range(25):
#     mean_std[i][0] = np.mean(np.abs(bone[:,:,:,i,:]))
#     mean_std[i][1] = np.std(np.abs(bone[:,:,:,i,:]))

# print(mean_std)
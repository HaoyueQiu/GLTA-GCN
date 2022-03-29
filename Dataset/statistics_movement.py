import numpy as np
from sklearn import preprocessing
paris = ((1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
        (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), 
        (11, 10), (12, 11), (13, 1),(14, 13), (15, 14), 
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
        (22, 23), (21, 21), (23, 8), (24, 25), (25, 12))

data_path = "/root/data/ntu_downsample150/xsub/train_data_joint.npy"
data = np.load(data_path)
N, C, T, V, M = data.shape
motion = data[:,:,1:,...]-data[:,:,:T-1,...]
mean_move = [[0 for i in range(25)]]
for i in range(25):
    mean_move[0][i] = np.mean(np.abs(motion[:,:,:,i,:]))
print(preprocessing.normalize(np.asarray(mean_move)))
# bone = np.zeros((N,C,T,V,M))
# for v1, v2 in paris:
#     v1 -= 1
#     v2 -= 1
#     bone[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]


# mean_std = [[0 for i in range(2)] for i in range(25)]
# for i in range(25):
#     mean_std[i][0] = np.mean(np.abs(bone[:,:,:,i,:]))
#     mean_std[i][1] = np.std(np.abs(bone[:,:,:,i,:]))

# print(mean_std)
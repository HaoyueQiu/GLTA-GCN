import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

alpha = [2,4,4,1]
flow1 = np.load('/root/Transformer_CV/customizeloss_shiftGCN_predict0.1_xsub.npz')
flow2 = np.load('/root/Transformer_CV/customizeloss_shiftGCN_rotation_xsub.npz')
flow3 = np.load('/root/Transformer_CV/customizeloss_shiftGCN_seg10_xsub.npz')
# flow4 = np.load('/root/Transformer_CV/rebuild.npz')
train_feature1 = preprocessing.normalize(flow1["train_feature"])
train_feature2 = preprocessing.normalize(flow2["train_feature"])
train_feature3 = preprocessing.normalize(flow3["train_feature"])
# train_feature4 = preprocessing.normalize(flow4["train_feature"])


test_feature1 = preprocessing.normalize(flow1["test_feature"])
test_feature2 = preprocessing.normalize(flow2["test_feature"])
test_feature3 = preprocessing.normalize(flow3["test_feature"])
# test_feature4 = preprocessing.normalize(flow4["test_feature"])

train_label = flow1["train_label"]  # train_label不同流不一样,所以is_test阶段时，dataset不要shuffle
test_label = flow1["test_label"]


def knn_accuracy(k,train_hidden_feature,train_labels,test_hidden_feature,test_labels):
    Xtr_Norm = preprocessing.normalize(train_hidden_feature)
    Xte_Norm = preprocessing.normalize(test_hidden_feature)

    knn = KNeighborsClassifier(n_neighbors=k,metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, train_labels)
    pred = knn.predict(Xte_Norm)
    accuracy = accuracy_score(pred, test_labels)
    return accuracy,pred

def get_confusion_matrix(pred,test_label):
    numclass = 60
    confusion_matrix = np.zeros((numclass,numclass))
    for i, l in enumerate(test_label):
        confusion_matrix[l, pred[i]] += 1
    return confusion_matrix

train_feature = np.concatenate((alpha[0]*train_feature1,alpha[1]*train_feature2,alpha[2]*train_feature3),axis=1)
test_feature = np.concatenate((alpha[0]*test_feature1,alpha[1]*test_feature2,alpha[2]*test_feature3),axis=1)



acc,pred = knn_accuracy(1,train_feature,train_label,test_feature,test_label)
confusion_matrix = get_confusion_matrix(pred,test_label)
print(acc)
print(confusion_matrix)
for i in range(60):
    class_acc = confusion_matrix[i][i]/np.sum(confusion_matrix[i])
    print(i,class_acc)
    for j in range(60):
        if j!=i and confusion_matrix[i][j]/np.sum(confusion_matrix[i]) > 0.1:
            print("*",j,confusion_matrix[i][j]/np.sum(confusion_matrix[i]))
    
# np.save('MSGCN_Confusionmatrix_NTURGBD_Xsub.npy',confusion_matrix)



# best_acc = 0

# for i in range(1,10):
#     for j in range(1,10-i):
#         alpha[0] = i
#         alpha[1] = j
#         alpha[2] = 10-i-j
#         train_feature = np.concatenate((alpha[0]*train_feature1,alpha[1]*train_feature2,alpha[2]*train_feature3),axis=1)
#         test_feature = np.concatenate((alpha[0]*test_feature1,alpha[1]*test_feature2,alpha[2]*test_feature3),axis=1)
#         acc = knn_accuracy(1,train_feature,train_label,test_feature,test_label)
#         print(i,j,10-i-j)
#         if acc>best_acc:
#             best_acc = acc
#             print("new best_acc",best_acc)


# train_feature = np.concatenate((1*train_feature1,4*train_feature2),axis=1)
# test_feature = np.concatenate((1*test_feature1,4*test_feature2),axis=1)
# # # train_feature = 3*train_feature1+4*train_feature2+3*train_feature3
# # # test_feature = 3*test_feature1+4*test_feature2+3*test_feature3
# acc = knn_accuracy(1,train_feature,train_label,test_feature,test_label)
# print(acc)


# for i in range(1,10):
#     for j in range(1,10):
#         alpha[0] = i
#         alpha[1] = j
#         train_feature = np.concatenate((alpha[0]*train_feature1,alpha[1]*train_feature2),axis=1)
#         test_feature = np.concatenate((alpha[0]*test_feature1,alpha[1]*test_feature2),axis=1)
#         acc = knn_accuracy(1,train_feature,train_label,test_feature,test_label)
#         print(i,j)
#         if acc>best_acc:
#             best_acc = acc
#             print("new best_acc",best_acc)


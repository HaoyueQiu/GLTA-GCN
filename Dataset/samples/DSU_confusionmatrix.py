import numpy as np

num_class= 4


fa = [i for i in range(num_class)]



def divide_class(confusion_matrix,score):
    """
    根据混淆矩阵决定两个节点是否属于难辩别的样本对
    """
    for i in range(num_class):
        for j in range(num_class):
            confusing_ratio = confusion_matrix[i][j]/confusion_matrix[i].sum()
            if confusing_ratio>score:
                merge(i,j)
    for i in range(num_class):
        find(i)
    ans = []
    for i in range(num_class):
        temp = []
        for j in range(num_class):
            if find(j) == i:
                temp.append(j)
        if temp:
            ans.append(temp)
    print(ans)

confusion_matrix = np.zeros((num_class,num_class))
confusion_matrix = np.asarray([[0,1,0,0],[1,2000,1,1],[1,1,4000,3],[4,5,600,1000]])
divide_class(confusion_matrix,0.02)

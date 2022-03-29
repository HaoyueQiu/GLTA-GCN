from DLNest.Common.DatasetBase import DatasetBase

import torch.utils.data as data
import torch
import numpy as np
import json
import pickle
import random

class NTUDataSet(data.Dataset):
    def __init__(self,arg:dict,is_test=False):
        """
        NTU RGBD数据集需要对每个骨架都进行非零帧补全、相对坐标等数据预处理操作，这些操作比较耗时，并且每次训练、测试网络时数据都是一致的。所以先统一数据预处理后再存入一个文件中(npy,pkl格式),而后每次构建dataset时只需要从这两个文件直接np.load、pickle.load即可，避免重复数据预处理。
        data_path：数据路径
        label_path: 标签路径
        """
        root = "/root/data/UWA3DTransformedPreprocess/"
        test = "3"
        if is_test==False:
            with open(root+"label_1.pkl",'rb') as f:
                label1 = pickle.load(f, encoding='latin1')
            with open(root+"label_2.pkl",'rb') as f:
                label2 = pickle.load(f, encoding='latin1')
            self.label = np.concatenate((label1,label2),axis=0)
        else:
            with open(root+"label_"+test+".pkl",'rb') as f:
                self.label = pickle.load(f, encoding='latin1')
        
        


        
        if is_test==False:
            data1 = np.load(root+"joint_1.npy", mmap_mode='r')
            data2 = np.load(root+"joint_2.npy", mmap_mode='r')
            self.data = np.concatenate((data1,data2),axis=0)
        else:
            self.data = np.load(root+"joint_"+test+".npy", mmap_mode='r')
            

        
        

    def __len__(self):
        return (len(self.label))

    def __getitem__(self,idx):
        # np.array和 np.asarray的区别 https://www.cnblogs.com/keye/p/11264599.html， np.array总是为深拷贝
        data = np.array(self.data[idx])
        label = self.label[idx]
        sample = dict(data=data,label=label,idx=idx)
        return sample
    


class Dataset(DatasetBase):
    def __init__(self,args : dict):
        """
        返回值
            dict,Dataloader,Dataloader
            dict: 告诉模型一些信息
            Dataloaders:训练和验证所用的dataloader，如果没有验证集则返回两个训练所用的dataloader
            Dataloaders for training,validation. If no validation, return two same dataloaders
        """
        self.args = args["dataset_config"]
        train_dataset_config = {}
        val_dataset_config = {}

        self.trainSet = NTUDataSet(train_dataset_config,False)
        self.valSet = NTUDataSet(val_dataset_config,True)

        def init_seed(_):
            torch.cuda.manual_seed_all(1)
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
           
        if args["is_test"]:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=False,
                num_workers=self.args['num_workers'],
                drop_last=False,
                worker_init_fn=init_seed
                )
        else:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=True,
                num_workers=self.args['num_workers'],
                drop_last=False,
                worker_init_fn=init_seed
                )
        self.valLoader = torch.utils.data.DataLoader(
            dataset=self.valSet,
            batch_size=self.args['batchsize'],
            shuffle=False,
            num_workers=self.args['num_workers'],
            drop_last=False,
            worker_init_fn=init_seed
        )
        


    def afterInit(self):
        return {},self.trainLoader,self.valLoader
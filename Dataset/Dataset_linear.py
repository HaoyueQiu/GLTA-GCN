from DLNest.Common.DatasetBase import DatasetBase

import torch.utils.data as data
import torch
import numpy as np
import json
import pickle
import random

class NTUDataSet(data.Dataset):
    def __init__(self,arg:dict):
        """
        NTU RGBD数据集需要对每个骨架都进行非零帧补全、相对坐标等数据预处理操作，这些操作比较耗时，并且每次训练、测试网络时数据都是一致的。所以先统一数据预处理后再存入一个文件中(npy,pkl格式),而后每次构建dataset时只需要从这两个文件直接np.load、pickle.load即可，避免重复数据预处理。
        data_path：数据路径
        label_path: 标签路径
        """
        if arg['use_mmap']:
            flow = np.load(arg['path'], mmap_mode='r')
        else:
            flow = np.load(arg['path'])
        if arg["is_test"]:
            self.data = flow["test_feature"]
            self.label = flow["test_label"]
        else:
            self.data = flow["train_feature"]
            self.label = flow["train_label"]

        if arg['debug']:
            self.label = self.label[0:100]
            self.data = self.data[0:100]

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
        train_dataset_config = {
            "path":self.args["directory"],
            "is_test":False,
            "use_mmap":self.args["use_mmap"],
            "debug":self.args["debug"]
        }
        val_dataset_config = {
            "path":self.args["directory"],
            "is_test":True,
            "use_mmap":self.args["use_mmap"],
            "debug":self.args["debug"]
        }

        self.trainSet = NTUDataSet(train_dataset_config)
        self.valSet = NTUDataSet(val_dataset_config)

        # 需要注意的是worker_init_fn
        # 它是worker初始化函数，非空的话会将worker id作为输入，在每次seeding和data loading之间调用。
        # 深度学习一个非常重要的问题就是结果复现问题，但shuffle会随机打乱数据，而数据放入模型训练的顺序会影响模型的最终效果，如何使其维持稳定呢？就需要初始化随机数种子。可以通过worker_init_fn这个参数运行初始化随机数种子的函数。
        
        def init_seed(_):
            torch.cuda.manual_seed_all(1)
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
            # torch.backends.cudnn.enabled = False
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        if args["is_test"]:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=False,
                num_workers=self.args['num_workers'],
                drop_last=True,
                worker_init_fn=init_seed
                )
        else:
            self.trainLoader = torch.utils.data.DataLoader(
                dataset=self.trainSet,
                batch_size=self.args['batchsize'],
                shuffle=True,
                num_workers=self.args['num_workers'],
                drop_last=True,
                worker_init_fn=init_seed
                )
        self.valLoader = torch.utils.data.DataLoader(
            dataset=self.valSet,
            batch_size=self.args['batchsize'],
            shuffle=False,
            num_workers=self.args['num_workers'],
            drop_last=True,
            worker_init_fn=init_seed
        )
        


    def afterInit(self):
        return {},self.trainLoader,self.valLoader
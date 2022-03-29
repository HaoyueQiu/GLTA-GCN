from DLNest.Common.ModelBase import ModelBase
import importlib
import numpy as np
from visdom import Visdom
import random
# torch
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from collections import OrderedDict
from Dataset.Dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from torch.nn.functional import kl_div as kld
import faiss
import time
import random
from torch import einsum
import os


class Model(ModelBase):
    def __init__(self,args : dict,datasetInfo : dict = None):
        self.log = {}
        # seed = random.randint(1,1000000)
        seed = 1
        self.init_seed(seed)
        print("randomseed",seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        self.total_epoch = args["epoch"]
        self.epoch = 0
        self.root_file_path = args["root_file_path"]
        self.args = args["model_config"]
        self.load_model(args)
        self.load_optimizer(args)
        if self.args["autoencoder"]:
            self.load_optimizer_autoencoder(args)
        
        # self.viz = Visdom(env='demo'+str(random.randint(1,1000000)))
        self.viz = Visdom(env='demo')
        self.viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc/iter', legend=['loss', 'acc']))
        self.viz.line([[0.,0.]], [0], win='test', opts=dict(title='mean_loss&acc/epoch', legend=['mean_loss', 'acc']))
        self.global_steps = 0
        self.trainLoader = Dataset(args).trainLoader

    
    def load_model(self,args:dict):

        output_device = self.args["device"][0] if type(self.args["device"]) is list else self.args["device"]
        self.output_device = output_device

        Model=getattr(importlib.import_module(self.args['model_path']),self.args['model_name'])
        self.model = Model(self.args["model_args"]["num_class"],
                            self.args["model_args"]["num_point"],
                            self.args["model_args"]["num_person"],
                            self.args["model_args"]["graph"],
                            self.args["model_args"]["graph_args"],
                            self.args["model_args"]["in_channels"],
                            self.args["model_args"]["self_supervised_mask"],
                            self.args["model_args"]["if_rotation"],
                            self.args["model_args"]["seg_num"],
                            self.args["model_args"]["if_vibrate"],
                            self.args["model_args"]["prediction_mask"],
                            self.args["model_args"]["GCNEncoder"],
                            self.args["model_args"]["ATU_layer"],
                            self.args["model_args"]["T"],
                            self.args["model_args"]["predict_seg"]
                            ).cuda(output_device)
        print(self.args["loss"])
        if self.args["loss"] == "CustomizeL2loss":
            CustomL2Loss=getattr(importlib.import_module(self.args['model_path']),'CustomizeL2Loss')
            self.loss = CustomL2Loss().cuda(output_device)
            
        elif self.args["loss"] == "L2loss":
            self.loss = nn.MSELoss().cuda(output_device)
        elif self.args["loss"] == "L1loss":
            self.loss = nn.L1Loss().cuda(output_device)
        else:
            self.loss = nn.SmoothL1Loss().cuda(output_device)
        
        if self.args["bone_length_loss"]:
            BoneLengthLoss = getattr(importlib.import_module(self.args['model_path']),'BoneLengthLoss')
            self.bone_lengthloss = BoneLengthLoss().cuda(output_device)
        
        if self.args["joint_direction_loss"]:
            JointDirectionPredictionLoss = getattr(importlib.import_module(self.args['model_path']),'JointDirectionPredictionLoss')
            self.joint_direction_loss = JointDirectionPredictionLoss().cuda(output_device)

        if self.args["motion_loss"]:
            MotionLoss = getattr(importlib.import_module(self.args['model_path']),'MotionLoss')
            self.motionloss = MotionLoss().cuda(output_device)
        
        if self.args["colorization"]:
            self.colorization_loss = nn.MSELoss().cuda()

        # self.custom_l2_loss = CustomL2Loss().cuda(output_device)

        if self.args["innerautoencoder_loss"] == True:
            self.innerautoencoder_loss = nn.MSELoss().cuda(output_device)

        if self.args["autoencoder"]:
            self.loss_autoencoder = nn.SmoothL1Loss().cuda(output_device)
            AutoEncoder=getattr(importlib.import_module(self.args['model_path']),'AutoEncoder')
            self.autoencoder = AutoEncoder().cuda(output_device)


        if type(self.args["device"]) is list:
            if len(self.args["device"]) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    # device_ids=[0,1,2,3],
                    device_ids=self.args["device"],
                    output_device=output_device) 
        self.is_test = args["is_test"]
        self.save_dir = args["save_dir"]
        if args["ckpt_load"]:
            save_dict = torch.load(args["ckpt_load"])
            self.loadSaveDict(save_dict)


    def load_optimizer(self,args:dict):
        if self.args["optimizer"] == 'SGD':
            params_dict = dict(self.model.named_parameters())
            params = []
            for key, value in params_dict.items():
                if self.args["innerautoencoder_loss"] and "autoEncoder" in key:
                    continue
                decay_mult = 0.0 if 'bias' in key else 1.0
                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                params += [{'params': value, 'lr': self.args["base_lr"], 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]
            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=True)
        elif self.args["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr= self.args["base_lr"],
                weight_decay= self.args["weight_decay"])
        else:
            raise ValueError()
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)
        if self.args["innerautoencoder_loss"] == True:
            params_dict = dict(self.model.named_parameters())
            params = []
            for key, value in params_dict.items():
                if "autoEncoder" not in key:
                    continue
                decay_mult = 0.0 if 'bias' in key else 1.0
                lr_mult = 1.0
                weight_decay = 1e-4
                if 'Linear_weight' in key:
                    weight_decay = 1e-3
                elif 'Mask' in key:
                    weight_decay = 0.0
                params += [{'params': value, 'lr': self.args["base_lr"], 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]
            
            self.innerautoencoder_optimizer =  optim.SGD(
                params,
                momentum=0.9,
                nesterov=True)
            

    def load_optimizer_autoencoder(self,args:dict):
        params_dict = dict(self.autoencoder.named_parameters())
        params = []
        for key, value in params_dict.items():
            decay_mult = 0.0 if 'bias' in key else 1.0
            lr_mult = 1.0
            weight_decay = 1e-4
            if 'Linear_weight' in key:
                weight_decay = 1e-3
            elif 'Mask' in key:
                weight_decay = 0.0
            params += [{'params': value, 'lr': self.args["base_lr"], 'lr_mult': lr_mult, 'decay_mult': decay_mult, 'weight_decay': weight_decay}]
        self.optimizer_autoencoder = optim.SGD(
            params,
            momentum=0.9,
            nesterov=True)
        self.lr_scheduler_autoencoder = ReduceLROnPlateau(self.optimizer_autoencoder, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)

    def initLog(self):
        log = {
            'one_step_log':{
                'epoch': 0,
                'iter': 0,
                'loss': 0.0,
                'acc': 0.0,
                'lr': 0.0,
                'network_time': 0.0
            },
            'validate':{
                'epoch':0,
                'accuracy':0.0,
                'best_acc':0.0,
                'mean_loss':0.0,
            }
        }
        self.log = log
        return self.log


    def getSaveDict(self):
        save_dict = {
            "weights": self.model.state_dict(),
        }
        return save_dict

    def loadSaveDict(self,saveDict : dict):
        output_device = self.args["device"][0] if type(self.args["device"]) is list else self.args["device"]
        weights = saveDict["weights"]
        self.epoch = saveDict["epoch"]
        
        # weights = [[k.split('module.')[-1],v.cuda(output_device)] for k, v in weights.items()]
        self.model.load_state_dict(weights)

    def runOneStep(self,data,log : dict,iter : int,epoch : int):
        if self.is_test:
            return
        self.model.train()
        sample = data
        data = Variable(sample['data'].float().cuda(self.output_device), requires_grad=False)
        if self.args["joint_direction_loss"]:
            label = self.DirectionPredictionLabel(data).cuda()
        # forward
        output,hidden_features,GCN_feature,autoencoder_output_feature,joint_direction_prediction = self.model(data)
        loss = self.args["joint_loss"]*self.loss(output[:,:3,...], data)
        if self.args["joint_direction_loss"]:
            loss += self.args["joint_direction_loss"] * self.joint_direction_loss(label,joint_direction_prediction)

        if self.args["bone_length_loss"]:
            loss += self.args["bone_length_loss"]*self.bone_lengthloss(data,output) #N,C,T,V,M
        if self.args["motion_loss"]:
            loss += self.args["motion_loss"]*self.motionloss(data,output)
        
        if self.args["colorization"]:
            colorization = Variable(sample['colorization'].float().cuda(self.output_device), requires_grad=False)
            loss += self.args["colorization"]*self.colorization_loss(colorization,output[:,3:,...])#N,C,T,V,M
        # custom_loss = self.custom_l2_loss(output,data)
        # print(loss,custom_loss)

        if self.args["R-drop"]:
            output1,hidden_features1,_ = self.model(data)
            # print("loss:",loss,end=" ")
            loss += self.loss(output1,data)
            # print(loss,end=" ")
            loss += self.args["R-drop-loss"]*(kld(output.softmax(dim=0).log(),output1.softmax(dim=0),reduction="batchmean")+kld(output1.softmax(dim=0).log(),output.softmax(dim=0),reduction="batchmean"))
            # print(loss)
        if self.args["R-rotation"]:
            output1,hidden_features1,_ = self.model(data)
            loss += self.loss(output1,data)
            loss += self.args["R-drop-loss"]*(kld(hidden_features.softmax(dim=0).log(),hidden_features1.softmax(dim=0),reduction="batchmean")+kld(hidden_features1.softmax(dim=0).log(),hidden_features.softmax(dim=0),reduction="batchmean"))


        # backward
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.args["innerautoencoder_loss"]:
            # backward
            innerautoencoder_loss = self.innerautoencoder_loss(GCN_feature,autoencoder_output_feature)
            self.innerautoencoder_optimizer.zero_grad()
            innerautoencoder_loss.backward()
            self.innerautoencoder_optimizer.step()
        
        self.optimizer.step()
        
        if self.args["autoencoder"]:
            self.autoencoder.train()
            input_autoencoder = Variable(hidden_features.data.float().cuda(self.output_device),requires_grad=False)
            output_autoencoder,hidden_features_autoencoder = self.autoencoder(input_autoencoder)
            loss_autoencoder = self.loss_autoencoder(output_autoencoder,input_autoencoder)
            self.optimizer_autoencoder.zero_grad()
            loss_autoencoder.backward()
            self.optimizer_autoencoder.step()
       
        self.log["one_step_log"]["epoch"] = epoch
        self.log["one_step_log"]["iter"] = iter
        self.log["one_step_log"]["loss"] = loss
        self.log["one_step_log"]["lr"] = self.optimizer.param_groups[0]['lr']
        self.global_steps +=1



    def visualize(self):
        loss = self.log["one_step_log"]["loss"]
        # acc = self.log["one_step_log"]["acc"]
        # self.viz.line([[loss.item(),acc.item()]], [self.global_steps], win='train', update='append')
        self.viz.line([loss.item()], [self.global_steps], win='train', update='append')

    def visualizePerEpoch(self):
        mean_loss = self.log["validate"]["mean_loss"]
        acc = self.log["validate"]["accuracy"]
        self.viz.line([[mean_loss,acc]], [self.epoch], win='test', update='append')

    def validate(self,valLoader,log : dict):
        start_time = time.time()
        self.adjust_learning_rate(self.epoch)
        self.model.eval()
        if self.args["autoencoder"]:
            self.autoencoder.eval()
        self.epoch +=1
        if self.epoch%self.args["test_epoch_interval"] !=0:
            return 0
        loss_value = []
        loss_value_autoencoder = []

        test_label_all = []

        train_hidden_features = []
        train_label_all =[]
        
        for batch_idx,sample in enumerate(self.trainLoader):
            with torch.no_grad():
                data = Variable(sample['data'].float().cuda(self.output_device), requires_grad=False)
                label = Variable(sample['label'].long().cuda(self.output_device), requires_grad=False)
                output,hidden_feature,GCN_feature,innerautoencoder_ouputfeature,joint_direction_prediction = self.model(data,True)
                if self.args["autoencoder"]:
                    put_autoencoder,hidden_feature_autoencoder = self.autoencoder(hidden_feature)

            if self.args["autoencoder"]:
                train_hidden_features.append(hidden_feature_autoencoder.detach().cpu().numpy())
            else:
                train_hidden_features.append(hidden_feature.detach().cpu().numpy())
            train_label_all.append(label.detach().cpu().numpy())

        train_label_all = np.concatenate(train_label_all)
        train_hidden_features = np.concatenate(train_hidden_features)
        

        test_hidden_features = []
        for batch_idx, sample in enumerate(valLoader):
            with torch.no_grad():
                data = Variable(sample['data'].float().cuda(self.output_device), requires_grad=False)
                label = Variable(sample['label'].long().cuda(self.output_device), requires_grad=False)
                if self.args["joint_direction_loss"]:
                    joint_direction_label = self.DirectionPredictionLabel(data).cuda()

                output,hidden_feature,GCN_feature,innerautoencoder_ouputfeature,joint_direction_prediction = self.model(data,True)
                if self.args["autoencoder"]:
                    output_autoencoder,hidden_feature_autoencoder = self.autoencoder(hidden_feature)

            loss =self.args["joint_loss"] * self.loss(output[:,:3,...], data)
            if self.args["colorization"]:
                colorization = Variable(sample['colorization'].float().cuda(self.output_device), requires_grad=False)
                loss += self.args["colorization"]*self.colorization_loss(colorization,output[:,3:,...])#N,C,T,V,M
                
            if self.args["joint_direction_loss"]:
                loss += self.args["joint_direction_loss"] * self.joint_direction_loss(joint_direction_label,joint_direction_prediction)


            if self.args["bone_length_loss"]:
                loss+=self.args["bone_length_loss"]*self.bone_lengthloss(data,output)


            if self.args["motion_loss"]:
                loss += self.args["motion_loss"]*self.motionloss(data,output)
            loss_value.append(loss.data.cpu().numpy())
            test_label_all.append(label.data.cpu().numpy())
            if self.args["autoencoder"]:
                test_hidden_features.append(hidden_feature_autoencoder.detach().cpu().numpy())
                loss_autoencoder = self.loss_autoencoder(output_autoencoder,hidden_feature)
                loss_value_autoencoder.append(loss_autoencoder.data.cpu().numpy())
            else:
                test_hidden_features.append(hidden_feature.detach().cpu().numpy())


        test_label_all = np.concatenate(test_label_all)
        test_hidden_features = np.concatenate(test_hidden_features)
        
        accuracy = self.knn_accuracy(self.args["knn_k"],train_hidden_features,train_label_all,test_hidden_features,test_label_all)
        end_time = time.time()
        if self.is_test:
            np.savez(self.save_dir,train_feature=train_hidden_features,test_feature=test_hidden_features,train_label=train_label_all,test_label=test_label_all)
            
        #log
        self.log["validate"]["test_time"] = end_time-start_time
        self.log["validate"]["epoch"] = self.epoch
        self.log["validate"]["mean_loss"] = np.mean(loss_value)
        self.log["validate"]["accuracy"] = accuracy
        if self.args["autoencoder"]:
            self.log["validate"]["autoencoder_loss"] = np.mean(loss_value_autoencoder)
        if accuracy > self.log["validate"]["best_acc"]:
            self.log["validate"]["best_acc"] = accuracy
        print(self.log["validate"])

    def knn_accuracy(self,k,train_hidden_feature,train_labels,test_hidden_feature,test_labels):

        Xtr_Norm = preprocessing.normalize(train_hidden_feature)
        Xte_Norm = preprocessing.normalize(test_hidden_feature)

        knn = KNeighborsClassifier(n_neighbors=k,metric='cosine')  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
        knn.fit(Xtr_Norm, train_labels)
        pred = knn.predict(Xte_Norm)
        accuracy = accuracy_score(pred, test_labels)
        return accuracy
    
    def knn_accuracy_faiss(self,k,train_hidden_feature,train_labels,test_hidden_feature,test_labels):
        self.index = faiss.IndexFlatL2(train_hidden_feature.shape[1])
        # self.index.add(train_hidden_feature.astype(np.float32))
        self.index.add(train_hidden_feature)
        distances, indices = self.index.search(test_hidden_feature.astype(np.float32), k=k)
        votes = train_labels[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        accuracy = accuracy_score(predictions, test_labels)
        return accuracy

    def adjust_learning_rate(self, epoch):
        lr = self.args["base_lr"] * (0.1 ** np.sum(epoch >= np.array(self.args["step"])))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.args["autoencoder"]:
            for param_group in self.optimizer_autoencoder.param_groups:
                param_group['lr'] = lr
        return lr

    def get_confusion_matrix(self,score,label):
        numclass = self.args["model_args"]["num_class"]
        confusion_matrix = np.zeros((numclass,numclass))
        rank = score.argsort()
        for i, l in enumerate(label):
            confusion_matrix[l, rank[i, -1]] += 1
        return confusion_matrix

    def test_info(self,confusion_matrix):

        numclass = self.args["model_args"]["num_class"]
        class_info_dict={
            "class_id":0,
            "class_name":"class_name",
            "samples_num":0,
            "class_accuracy":0.0,
            "top_5":[['action1',0],['action2',0],['action3',0],['action4',0],['action5',0]]
        }

        info = []
        for i in range(numclass):
            class_info_dict["class_id"] = i  
            class_info_dict["class_name"] = NTURGBD_CLASS_NAME_ID[i]      
            class_info_dict["samples_num"] = confusion_matrix[i].sum()
            class_info_dict["class_accuracy"] = confusion_matrix[i][i]/class_info_dict["samples_num"]
            class_info_dict["top_5"] = [[l,NTURGBD_CLASS_NAME_ID[l],confusion_matrix[i][l]/class_info_dict["samples_num"]] for l in confusion_matrix[i].argsort()[-5:][::-1]]
            info.append(class_info_dict.copy())
        info.sort(key=lambda x:x["class_accuracy"])
        for i in info:
            # print(i)
            print("class_id: {0:<3}, class_name: {1:<50}, samples_num: {2:<10}, class accuracy:{3:<0.3f}".format(i["class_id"],i["class_name"],i["samples_num"],i["class_accuracy"]),end=" ")
            for j in range(len(i["top_5"])):
                print("top5:[id:{0:<3},name:{1:<50},confusion_rate:{2:0.2f}]".format(i["top_5"][j][0],i["top_5"][j][1],i["top_5"][j][2]),end=" ")
            print()
        
        self.get_difficult_class(confusion_matrix,0.03)

        # for i in info:     

    def top_k(self, score, label,top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_difficult_class(self,confusion_matrix,score):
        num_class = self.args["model_args"]["num_class"]
        fa = [i for i in range(num_class)]
        for i in range(num_class):
            for j in range(num_class):
                confusing_ratio = confusion_matrix[i][j]/confusion_matrix[i].sum()
                if confusing_ratio>score:
                    self.merge(fa,i,j)
        for i in range(num_class):
            self.find(fa,i)
        ans = []
        for i in range(num_class):
            temp = []
            for j in range(num_class):
                if self.find(fa,j) == i:
                    temp.append(j)
            if temp:
                ans.append(temp)
        print(ans)

    def find(self,fa,x):
        if fa[x] == x:
            return x
        else:
            fa[x] = self.find(fa,fa[x])
            return fa[x]

    def merge(self,fa,i,j):
        fa[self.find(fa,i)] = self.find(fa,j)

    def init_seed(self,seed):
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
    
    def joint_label(self,joint_motion):
        label = 0
        for i in joint_motion:
            label = label*3
            if i == 0.0:
                label +=1
            elif i==1.0:
                label +=2
        return label

    def DirectionPredictionLabel(self,data):
        N,C,T,V,M = data.shape
        label = torch.zeros((N,V,27))
        for i in range(len(data)):
            t = random.randint(1,149)
            motion = torch.sign(data[i,:,t,:,0]-data[i,:,t-1,:,0]) #C,V
            data[i,:,t-1,:,:] = 0
            for j,joint_motion in enumerate(motion.T):
                    L = self.joint_label(joint_motion)
                    label[i,j,L] = 1
        return label

NTURGBD_CLASS_NAME_ID = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading', 'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap', 'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping (one foot jumping)', 'jump up', 'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard', 'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute', 'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough', 'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)', 'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person', 'kicking other person', 'pushing other person', 'pat on back of other person', 'point finger at the other person', 'hugging other person', 'giving something to other person', "touch other person's pocket", 'handshaking', 'walking towards each other', 'walking apart from each other', 'put on headphone', 'take off headphone', 'shoot at the basket', 'bounce ball', 'tennis bat swing', 'juggling table tennis balls', 'hush (quite)', 'flick hair', 'thumb up', 'thumb down', 'make ok sign', 'make victory sign', 'staple book', 'counting money', 'cutting nails', 'cutting paper (using scissors)', 'snapping fingers', 'open bottle', 'sniff (smell)', 'squat down', 'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face', 'apply cream on hand back', 'put on bag', 'take off bag', 'put something into a bag', 'take something out of a bag', 'open a box', 'move heavy objects', 'shake fist', 'throw up cap/hat', 'hands up (both hands)', 'cross arms', 'arm circles', 'arm swings', 'running on the spot', 'butt kicks (kick backward)', 'cross toe touch', 'side kick', 'yawn', 'stretch oneself', 'blow nose', 'hit other person with something', 'wield knife towards other person', 'knock over other person (hit with body)', 'grab other person’s stuff', 'shoot at other person with a gun', 'step on foot', 'high-five', 'cheers and drink', 'carry something with other person', 'take a photo of other person', 'follow other person', 'whisper in other person’s ear', 'exchange things with other person', 'support somebody with hand', 'finger-guessing game (playing rock-paper-scissors)']

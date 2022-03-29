from DLNest.Common.DatasetBase import DatasetBase
from DLNest.Common.ModelBase import ModelBase
from DLNest.Common.LifeCycleBase import LifeCycleBase


class LifeCycle(LifeCycleBase):
    def __init__(self,model : ModelBase = None,dataset : DatasetBase = None, trainProcess = None, analyzeProcess = None):
        self.model = model
        self.dataset = dataset
        self.trainProcess = trainProcess
        self.analyzeProcess = analyzeProcess

    def needVisualize(self, epoch : int, iter : int, logdict : dict, args : dict):
        if self.model.is_test:
            return False
        else:
            return True

    def needValidation(self, epoch : int, logdict : dict, args : dict):
        return True

    def commandLineOutput(self,epoch : int, logdict : dict, args : dict):
        print("Epoch #" + str(self.model.epoch) + " finished!")

    def needSaveModel(self, epoch : int, logdict : dict, args : dict):
        return True

    def holdThisCheckpoint(self, epoch : int, logdict : dict, args : dict):
        #持久通道，自行决定是否保留当前checkpoint
        # 我用于保留最好的模型，将该通道大小设置为1，即可保留最佳模型
        if logdict["validate"]["best_acc"] == logdict["validate"]["accuracy"]:
            return True
        else:
            return False

    def needContinueTrain(self, epoch : int, logdict : dict, args : dict):
        if self.model.epoch < args['epoch'] and not self.model.is_test:
            return True
        else:
            return False

    def AOneEpoch(self):
        # 每过一个epoch调用
        self.model.visualizePerEpoch()
def experience(self):
    print(self.args,self.model,self.dataset)
    self.model.epoch = self.model.total_epoch
    print(self.model.epoch)
    print(self.model.total_epoch)
    self.model.validate(self.dataset.valLoader,{})
    
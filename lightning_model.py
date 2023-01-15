import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc
from scheduler import CosineAnnealingWarmUpRestarts
import torch
import torch.nn as nn
import torchvision.models as models

class MRSClassfication2D(pl.LightningModule):

    def __init__(self,model_config, class_weights):
        super().__init__()
        self.save_hyperparameters()
        if class_weights is not None:
            self.class_weights =torch.as_tensor(class_weights,dtype=torch.float)
        else :
            self.class_weights =class_weights
        print("class weights : ",self.class_weights)
        
        self.config = model_config
        self.num_classes = self.config['model_params']['cls_num_classes']
        self.model = self.create_model(self.config['model_name'],self.num_classes)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.num_classes = self.config['model_params']['cls_num_classes']

    def create_model(self,model_name, num_classes):
        model = getattr(models ,model_name)(weights ='DEFAULT')
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, num_classes)
        return model
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        pred = self(x)
        loss = self.loss (pred, y)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':loss,'pred':pred,'label':y}
        
        return output
    
    def training_epoch_end(self, outputs):
       if self.trainer.num_devices > 1:
           outputs = self.all_gather(outputs)
       
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       f1_macro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='micro')
       
       self.log("train_auc", auc,prog_bar=True, logger=True)
       self.log("train_f1", f1_macro,prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        pred = self(x)
        loss = self.loss (pred, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':loss,'pred':pred,'label':y}

        return output

    def validation_epoch_end(self, outputs):
        
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        
        preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
        labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)

        auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
        f1_macro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='micro')
        
        self.log("val_auc", auc, prog_bar=True, logger=True,on_epoch=True)
        self.log("val_f1", f1_macro,prog_bar=True, logger=True)
        self.log("mean_f1_auc", (auc+f1_macro)/2, prog_bar=True, logger=True,on_epoch=True)
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, 
                                                  T_0=self.config['T_0'], 
                                                  T_mult=self.config['T_mult'], 
                                                  eta_max=self.config['eta_max'],
                                                  T_up=self.config['T_up'],
                                                  gamma=self.config['gamma'])
        return [optimizer],[scheduler]


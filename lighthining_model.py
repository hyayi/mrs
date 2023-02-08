import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc
import models
import optimizers
import schedulers

class MRSClassficationMultiModal(pl.LightningModule):

    def __init__(self,model_config, class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.config = model_config
        
        if class_weights is not None:
            self.class_weights =torch.as_tensor(class_weights,dtype=torch.float)
        else :
            self.class_weights =class_weights
            
        print("class weights : ",self.class_weights)
        
        self.num_classes = self.config['model']['params']['num_classes']
        self.model = getattr(models,self.config['model']['name'])(**self.config['model']['params'])
        self.clsloss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self(img)
        
        loss = self.clsloss(pred, label)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'loss':loss,'pred':pred,'label':label}
        
        return output
    
    def training_epoch_end(self, outputs): 
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
        
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       f1_micro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='micro')
       f1_macro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='macro')
       
       self.log("train_auc", auc,prog_bar=False, logger=True)
       self.log("train_f1_micro", f1_micro,prog_bar=False, logger=True)
       self.log("train_f1_macro", f1_macro,prog_bar=False, logger=True)

    def validation_step(self, batch, batch_idx):
        
        img, label = batch
        pred = self(img)
        
        loss = self.clsloss(pred, label)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        output = {'loss':loss,'pred':pred,'label':label}

        return output

    def validation_epoch_end(self, outputs):
        
        if self.trainer.num_nodes > 1:
            outputs = self.all_gather(outputs)
        
        preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
        labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)

        auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
        f1_micro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='micro')
        f1_macro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='macro')
        
        self.log("val_auc", auc, prog_bar=True, logger=True,on_epoch=True)
        self.log("val_f1_micro", f1_micro,prog_bar=True, logger=True)
        self.log("val_f1_macro", f1_macro,prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        
        img, label = batch
        pred = self(img)
        output = {'pred':pred,'label':label}

        return output

    def test_epoch_end(self, outputs):
        
        if self.trainer.num_nodes > 1:
            outputs = self.all_gather(outputs)
        
        preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
        labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)

        auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
        f1_micro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='micro')
        f1_macro = multiclass_f1_score(preds,labels,num_classes=self.num_classes, average='macro')
        
        self.log("test_auc", auc, prog_bar=True, logger=True,on_epoch=True)
        self.log("test_f1_micro", f1_micro,prog_bar=True, logger=True)
        self.log("test_f1_macro", f1_macro,prog_bar=True, logger=True)
               
    def configure_optimizers(self):
        optimizer = getattr(optimizers,self.config['optimizer']['name'])(self.parameters(), **self.config['optimizer']['params'])
        scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer],[scheduler]

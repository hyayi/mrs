import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc,confusion_matrix
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

    def forward(self, img, clinical):
        return self.model(img, clinical)

    def training_step(self, batch, batch_idx):
        img, clinical, label = batch
        img_pred, clinical_pred, pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = 0.25*img_loss + 0.25*clinical_loss + 0.5*loss
        
        self.log("train_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':total_loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,  "label" :label}
        
        return output
    
    def training_epoch_end(self, outputs): 
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
       clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       
       self.log_dict({"train_auc" :  auc, "train_img_auc" : img_auc, "train_clinical_auc" : clinical_auc}, prog_bar=True, logger=True)
       

    def validation_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        
        self.log("val_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,  "label" :label}

        return output

    def validation_epoch_end(self, outputs):
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
           
       img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
       clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       
       self.log_dict({"val_auc" :  auc, "val_img_auc" : img_auc, "val_clinical_auc" : clinical_auc}, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred = self(img, clinical)
        output = {'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,  "label" :label}
        
        return output

    def test_epoch_end(self, outputs):
        
        if self.trainer.num_nodes > 1:
            outputs = self.all_gather(outputs)
        
        img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
        clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
        preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
        labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
        img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
        clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
        auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
        confusion_matrix_value = confusion_matrix(preds,labels,num_classes=self.num_classes,task="multiclass")
        
        self.log("auc", auc, prog_bar=True, logger=True,on_epoch=True)
        self.log("img_auc", img_auc, prog_bar=True, logger=True,on_epoch=True)
        self.log("clinical_auc", clinical_auc, prog_bar=True, logger=True,on_epoch=True)
        self.log('TN',confusion_matrix_value[0][0])
        self.log('FP',confusion_matrix_value[0][1])
        self.log('FN',confusion_matrix_value[1][0])
        self.log('TP',confusion_matrix_value[1][1])
        
    def configure_optimizers(self):
        optimizer = getattr(optimizers,self.config['optimizer']['name'])(self.parameters(), **self.config['optimizer']['params'])
        scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer],[scheduler]

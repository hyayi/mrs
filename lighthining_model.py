import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc,confusion_matrix
import models
import optimizers
import schedulers
class MRSClassficationMultiModal1(pl.LightningModule):

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
        if model_config['model']['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, img, clinical):
        return self.model(img, clinical)

    def training_step(self, batch, batch_idx):
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = self.clsloss(mean_pred, label)
        
        self.log("train_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':total_loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,'mean_pred' : mean_pred, "label" :label}
        
        return output
    
    def training_epoch_end(self, outputs): 
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       mean_pred = torch.cat([x['mean_pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
       clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       mean_auc = auroc(mean_pred, labels, task='multiclass', num_classes=self.num_classes)
       
       self.log_dict({"train_concat_auc" :  auc, "train_img_auc" : img_auc, "train_clinical_auc" : clinical_auc, "train_auc" : mean_auc}, prog_bar=True, logger=True)
       

    def validation_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = self.clsloss(mean_pred, label)
        
        self.log("val_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':total_loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,'mean_pred' : mean_pred, "label" :label}
        
        return output

    def validation_epoch_end(self, outputs):
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       mean_pred = torch.cat([x['mean_pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
       clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       mean_auc = auroc(mean_pred, labels, task='multiclass', num_classes=self.num_classes)
       
       self.log_dict({"val_concat_auc" :  auc, "val_img_auc" : img_auc, "val_clinical_auc" : clinical_auc, "val_auc" : mean_auc}, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        output = {'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred, 'mean_pred' : mean_pred,   "label" :label}
        
        return output

    def test_epoch_end(self, outputs):
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'].view(-1,self.num_classes) for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'].view(-1,self.num_classes) for x in outputs])
       preds = torch.cat([x['pred'].view(-1,self.num_classes) for x in outputs])
       mean_pred = torch.cat([x['mean_pred'].view(-1,self.num_classes) for x in outputs])
       labels = torch.cat([x['label'].view(-1) for x in outputs]).view(-1)
       
       img_auc = auroc(img_preds, labels, task='multiclass', num_classes=self.num_classes)
       clinical_auc = auroc(clinical_preds, labels, task='multiclass', num_classes=self.num_classes)
       auc = auroc(preds, labels, task='multiclass', num_classes=self.num_classes)
       mean_auc = auroc(mean_pred, labels, task='multiclass', num_classes=self.num_classes)
       
       self.log_dict({"test_concat_auc" :  auc, "test_img_auc" : img_auc, "test_clinical_auc" : clinical_auc, "test_auc" : mean_auc}, prog_bar=True, logger=True)

        
    def configure_optimizers(self):
        optimizer = getattr(optimizers,self.config['optimizer']['name'])(self.parameters(), **self.config['optimizer']['params'])
        #scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer]#,[scheduler]
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
        if model_config['model']['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, img, clinical):
        return self.model(img, clinical)

    def training_step(self, batch, batch_idx):
        img, clinical, label = batch
        img_pred, clinical_pred, pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = (img_loss + clinical_loss + loss)/3
        
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
        if model_config['model']['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, img, clinical):
        return self.model(img, clinical)

    def training_step(self, batch, batch_idx):
        img, clinical, label = batch
        img_pred, clinical_pred, pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = (img_loss + clinical_loss + loss)/3
        
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
        #scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer]


class MRSClassficationMultiModalBE(pl.LightningModule):

    def __init__(self,model_config, class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.config = model_config

            
        
        self.num_classes = self.config['model']['params']['num_classes']
        self.model = getattr(models,self.config['model']['name'])(**self.config['model']['params'])
        self.clsloss = nn.BCEWithLogitsLoss()
        if model_config['model']['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, img, clinical):
        return self.model(img, clinical)

    def training_step(self, batch, batch_idx):
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = self.clsloss(mean_pred, label)
        
        self.log("train_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':total_loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,'mean_pred' : mean_pred, "label" :label}
        
        return output
    
    def training_epoch_end(self, outputs): 
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'] for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'] for x in outputs])
       preds = torch.cat([x['pred'] for x in outputs])
       mean_pred = torch.cat([x['mean_pred'] for x in outputs])
       labels = torch.cat([x['label'] for x in outputs])
       
       img_auc = auroc(img_preds, labels, task='binary')
       clinical_auc = auroc(clinical_preds, labels, task='binary')
       auc = auroc(preds, labels, task='multiclbinaryass')
       mean_auc = auroc(mean_pred, labels, task='binary')
       
       self.log_dict({"train_concat_auc" :  auc, "train_img_auc" : img_auc, "train_clinical_auc" : clinical_auc, "train_auc" : mean_auc}, prog_bar=True, logger=True)
       

    def validation_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        
        img_loss = self.clsloss(img_pred, label)
        clinical_loss = self.clsloss(clinical_pred, label)
        loss = self.clsloss(pred, label)
        total_loss = self.clsloss(mean_pred, label)
        
        self.log("val_img_loss", img_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_clinical_loss", clinical_loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_total_loss", total_loss, on_epoch=True, prog_bar=True, logger=True)
        
        output = {'loss':total_loss,'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred,'mean_pred' : mean_pred, "label" :label}
        
        return output

    def validation_epoch_end(self, outputs):
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'] for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'] for x in outputs])
       preds = torch.cat([x['pred'] for x in outputs])
       mean_pred = torch.cat([x['mean_pred'] for x in outputs])
       labels = torch.cat([x['label'] for x in outputs])
       
       img_auc = auroc(img_preds, labels, task='binary')
       clinical_auc = auroc(clinical_preds, labels, task='binary')
       auc = auroc(preds, labels, task='multiclbinaryass')
       mean_auc = auroc(mean_pred, labels, task='binary')
       
       self.log_dict({"val_concat_auc" :  auc, "val_img_auc" : img_auc, "val_clinical_auc" : clinical_auc, "val_auc" : mean_auc}, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        
        img, clinical, label = batch
        img_pred, clinical_pred, pred, mean_pred = self(img, clinical)
        output = {'pred':pred,'img_pred' : img_pred, 'clinical_pred' : clinical_pred, 'mean_pred' : mean_pred,   "label" :label}
        
        return output

    def test_epoch_end(self, outputs):
        
       if self.trainer.num_nodes > 1:
           outputs = self.all_gather(outputs)
       img_preds = torch.cat([x['img_pred'] for x in outputs])
       clinical_preds = torch.cat([x['clinical_pred'] for x in outputs])
       preds = torch.cat([x['pred'] for x in outputs])
       mean_pred = torch.cat([x['mean_pred'] for x in outputs])
       labels = torch.cat([x['label'] for x in outputs])
       
       img_auc = auroc(img_preds, labels, task='binary')
       clinical_auc = auroc(clinical_preds, labels, task='binary')
       auc = auroc(preds, labels, task='multiclbinaryass')
       mean_auc = auroc(mean_pred, labels, task='binary')
       
       self.log_dict({"test_concat_auc" :  auc, "test_img_auc" : img_auc, "test_clinical_auc" : clinical_auc, "test_auc" : mean_auc}, prog_bar=True, logger=True)

        
    def configure_optimizers(self):
        optimizer = getattr(optimizers,self.config['optimizer']['name'])(self.parameters(), **self.config['optimizer']['params'])
        #scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer]#,[scheduler]
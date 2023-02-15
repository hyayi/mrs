import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc
import torch
import torch.nn as nn
import torchvision.models as models
import optimizers
import schedulers


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
        self.num_classes = self.config['model']['params']['num_classes']
        self.model = self.create_model(self.config['model']['name'],self.num_classes)
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def create_model(self,model_name, num_classes):
        if model_name == 'rensnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, num_classes)
        elif model_name == 'inceptionv_v3':
            model = models.inception_v3(pretrained=True)
            model.AuxLogits.fc = nn.Linear(768, num_classes)
            model.fc = nn.Linear(2048, num_classes)
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
       if self.trainer.num_nodes > 1:
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
        
    def test_step(self, batch, batch_idx):
        
        x, y = batch
        pred = self(x)
        output = {'pred':pred,'label':y}

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
        print(self.parameters())
        optimizer = getattr(optimizers,self.config['optimizer']['name'])(self.parameters(), **self.config['optimizer']['params'])
        scheduler = getattr(schedulers,self.config['scheduler']['name'])(optimizer,**self.config['scheduler']['params'])
        
        return [optimizer],[scheduler]


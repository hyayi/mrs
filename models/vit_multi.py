import torch.nn as nn
import torchvision
import torch
from monai.networks.nets import ViT 

class VitMultiModalFeatureConcat(nn.Module):
    def __init__(self, clinical_feature_len,head='linear', num_classes=2,img_size=(64,256,256), patch_size=(8,32,32) ) -> None:
        super().__init__()
        
        self.backbone = ViT(in_channels=1, img_size=img_size,patch_size=patch_size, pos_embed='conv')
        
        if head == 'linear':
            self.head = nn.Linear(768+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(768+clinical_feature_len, int((768+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((768+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x,_ = self.backbone(img)
        x = torch.cat([x[:,0], clinical], dim=1)
        x = self.head(x)
        return x
    

class VitMultiModalProbConcat(nn.Module):
    def __init__(self, clinical_feature_len,head='linear', num_classes=2,img_size=(64,256,256), patch_size=(8,32,32) ) -> None:
        super().__init__()
        
        self.backbone = ViT(in_channels=1, img_size=img_size,patch_size=patch_size, pos_embed='conv',classification=True)
        
        if head == 'linear':
            self.head = nn.Linear(clinical_feature_len+1, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(num_classes+clinical_feature_len, int((num_classes+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((num_classes+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x,_ = self.backbone(img)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class VitMultiModalPaper(nn.Module):
    def __init__(self, clinical_feature_len,
                 hidden_dims_1,
                 hidden_dims_2,
                 hidden_dims_3,
                 drop_out_rate_1,
                 drop_out_rate_2,
                 drop_out_rate_3,
                 num_classes=2,
                 img_size=(64,256,256), 
                 patch_size=(8,32,32) ) -> None:
        super().__init__()
        
        self.backbone = ViT(in_channels=1, img_size=img_size,patch_size=patch_size, pos_embed='conv',classification=False)
        
        self.fc1 = nn.Sequential(nn.Linear(768, hidden_dims_1[0]), 
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(drop_out_rate_1),
                                 nn.Linear(hidden_dims_1[0], hidden_dims_1[1]),
                                 nn.ReLU(inplace=True))
        
        self.fc2 = nn.Sequential(
                            nn.Linear(clinical_feature_len, hidden_dims_2[0]), 
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dims_2[0],hidden_dims_2[1]), 
                            nn.ReLU(inplace=True),
                            nn.Dropout(drop_out_rate_2),
                            nn.Linear(hidden_dims_2[1], hidden_dims_2[2]),
                            nn.ReLU(inplace=True))


        self.head =  nn.Sequential(
                            nn.Linear(
                                hidden_dims_1[1]+hidden_dims_2[2],
                                hidden_dims_3[0]), 
                            nn.ReLU(inplace=True),
                            nn.Dropout(drop_out_rate_3),
                            nn.Linear(hidden_dims_3[0], num_classes))
        
    
    def forward(self, img, clinical):
        img,_ = self.backbone(img)
        img = self.fc1(img[:,0])
        clinical =  self.fc2(clinical)
        out = self.head(torch.cat([img, clinical], dim=1))
        return out


class VitMultiModalEnsenble(nn.Module):
    def __init__(self, clinical_feature_len,head='linear', num_classes=2,img_size=(64,256,256), patch_size=(8,32,32) ) -> None:
        super().__init__()
        
        self.backbone = ViT(in_channels=1, img_size=img_size,patch_size=patch_size, pos_embed='conv',classification=True)
        
        if head == 'linear':
            self.head = nn.Linear(clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(clinical_feature_len, int((clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        img,_ = self.backbone(img)
        clinical = self.head(clinical)
        out = (img + clinical)/2
        return out
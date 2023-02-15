import torch.nn as nn
import torchvision
import torch
import torchvision.models as models

class Resnet50MultiModalFeatureConcat(nn.Module):
    def __init__(self, clinical_feature_len,head='linear', num_classes=2) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        if head == 'linear':
            self.head = nn.Linear(2048+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(2048+clinical_feature_len, int((2048+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((2048+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class Resnet50MultiModalProbConcat(nn.Module):
    def __init__(self, clinical_feature_len, head='linear', num_classes=2) -> None:
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)
        
        if head == 'linear':
            self.head = nn.Linear(clinical_feature_len+num_classes, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(num_classes+clinical_feature_len, int((num_classes+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((num_classes+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class Resnet50MultiModalEnsenble(nn.Module):
    def __init__(self, clinical_feature_len, head='linear', num_classes=2) -> None:
        super().__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, num_classes)
        
        if head == 'linear':
            self.head = nn.Linear(clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(clinical_feature_len, int((clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        img = self.backbone(img)
        clinical = self.head(clinical)
        out = (img + clinical)/2
        return out
    
    
class Resnet50MultiModalPaper(nn.Module):
    def __init__(self, 
                 clinical_feature_len,
                 hidden_dims_1,
                 hidden_dims_2,
                 hidden_dims_3,
                 drop_out_rate_1,
                 drop_out_rate_2,
                 drop_out_rate_3,
                 num_classes=2) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        
        self.fc1 = nn.Sequential(nn.Linear(2048, hidden_dims_1[0]), 
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
        img = self.backbone(img)
        img = img.flatten(start_dim=1)
        img = self.fc1(img)
        clinical =  self.fc2(clinical)
        out = self.head(torch.cat([img, clinical], dim=1))
        return out
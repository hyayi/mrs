import torch.nn as nn
import torchvision
import torch

class R2plus1d18MultiModalFeatureConcat(nn.Module):
    def __init__(self, clinical_feature_len,head='linear', num_classes=2) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(*list(torchvision.models.video.r2plus1d_18(pretrained=True).children())[:-1])
        self.backbone[0][0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        
        if head == 'linear':
            self.head = nn.Linear(512+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(512+clinical_feature_len, int((512+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((512+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class R2plus1d18MultiModalProbConcat(nn.Module):
    def __init__(self, clinical_feature_len , head='linear', num_classes=2) -> None:
        super().__init__()
        self.backbone = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.backbone.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.backbone.fc = nn.Linear(512, num_classes)
        
        if head == 'linear':
            self.head = nn.Linear(clinical_feature_len+1, num_classes)
    
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
    
class R2plus1d18MultiModalEnsenble(nn.Module):
    def __init__(self, clinical_feature_len, head='linear', num_classes=2) -> None:
        super().__init__()
        
        self.backbone = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.backbone.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.backbone.fc = nn.Linear(512, num_classes)
        
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

class R2plus1MultiModalPaper(nn.Module):
    def __init__(self, clinical_feature_len, num_classes=2) -> None:
        super().__init__()
        
        self.backbone = nn.Sequential(*list(torchvision.models.video.r2plus1d_18(pretrained=True).children())[:-1])
        self.backbone[0][0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256), 
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 128),
                                 nn.ReLU(inplace=True))
        
        self.fc2 = nn.Sequential(
                            nn.Linear(clinical_feature_len, 11), 
                            nn.ReLU(inplace=True),
                            nn.Linear(11,10), 
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.2),
                            nn.Linear(10, 10),
                            nn.ReLU(inplace=True))


        self.head =  nn.Sequential(
                            nn.Linear(138,60), 
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.2),
                            nn.Linear(60, num_classes))
        
    
    def forward(self, img, clinical):
        img = self.backbone(img)
        img = img.flatten(start_dim=1)
        clinical =  self.fc2(clinical)
        out = self.head(torch.cat([img, clinical], dim=1))
        return out

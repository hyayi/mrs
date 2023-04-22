import torch.nn as nn
import torchvision
from utils import create_prameter
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
import torch
from monai.networks.nets import ViT 

class R2plus1d_18(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super().__init__()
        self.model = torchvision.models.video.r2plus1d_18(pretrained=True)
        self.model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x
    

class Mc3_18(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super().__init__()
        self.model =torchvision.models.video.mc3_18(pretrained=True)
        self.model.stem[0] = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.model(x)
        return x

class nnUnet(Generic_UNet):
    """Training with image only"""

    def __init__(self, plans_path,num_classes=2, weight=None, return_features=False):
        
        parameter = create_prameter(plans_path)
        super().__init__(**parameter)
        self.retrun_features = return_features
        if weight :
            self.load_state_dict(torch.load(weight)['state_dict'],strict=False)
        
        self.fc = nn.Linear(320, num_classes)
    
    def forward(self,x):

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        img_em = F.adaptive_avg_pool3d(x, (1, 1,1)).view(x.shape[0],-1)
        out = self.fc(img_em)
        if self.retrun_features:
            return out,img_em
        return out
    
class nnUnetGL(Generic_UNet):
    """Training with image only"""

    def __init__(self, plans_path,num_classes=2, weight=None,fc_list = None):
        
        parameter = create_prameter(plans_path)
        super().__init__(**parameter)

        if weight :
            self.load_state_dict(torch.load(weight)['state_dict'],strict=False)
        
        layers = []
        in_features = 320 
        
        for i in range(len(fc_list)):
            
            out_features = fc_list[i]
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features
            
        layers.append(nn.Linear(in_features, num_classes))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self,x):

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        img_em = F.adaptive_avg_pool3d(x, (1, 1,1)).view(x.shape[0],-1)
        out = self.fc(img_em)
   
        return out
class Vit(nn.Module):
    def __init__(self,num_classes=2,img_size=(64,256,256), patch_size=(8,32,32) ) -> None:
        super().__init__()
        
        self.backbone = ViT(in_channels=1, img_size=img_size,num_classes=num_classes,patch_size=patch_size, pos_embed='conv',classification=True)
        
    def forward(self, img):
        x , _ = self.backbone(img)
        return x

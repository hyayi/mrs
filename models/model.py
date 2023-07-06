import torch.nn as nn
import torchvision
from utils import create_prameter
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
import torch
from monai.networks.nets import ViT,EfficientNetBN

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

    def __init__(self, plans_path,num_classes=2, weight=None):
        
        parameter = create_prameter(plans_path)
        super().__init__(**parameter)
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
        return  out
    
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
        
        self.backbone = ViT(in_channels=1, img_size=img_size,num_classes=num_classes,patch_size=patch_size, pos_embed='conv',classification=True, post_activation=None)
        
    def forward(self, img):
        x , _ = self.backbone(img)
        return x
        
class EfficientNet(nn.Module):
    def __init__(self, model_name = "efficientnet-b2" ,pretrained=False,spatial_dims=3, in_channels= 1, num_classes = 2):
        super().__init__()
        self.backbone = EfficientNetBN( model_name = model_name ,pretrained=pretrained,spatial_dims=spatial_dims, in_channels= in_channels, num_classes =num_classes)
    def forward(self,img):
        x = self.backbone(img)
        return x
        
class Resnet50(nn.Module):
    def __init__(self,input_chanel=1,pretrained=True,num_class=2):
        super().__init__()
        self.model_ft = models.video.r3d_18(pretrained=pretrained)
        prev_w = self.model_ft.stem[0].weight
        self.model_ft.stem[0] = nn.Conv3d(input_chanel, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.model_ft.stem[0].weight.data = prev_w.data.sum(dim=1 ,keepdim=True)
        self.model_ft.fc = nn.Linear(in_features=512, out_features=num_class, bias=True)
    
    def forward(self,x):
        out = self.model_ft(x)
        return out 

import torch 
import pickle
import torch.nn as nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F
from .d1cnn import SoftOrdering1DCNN

def create_layers(layer_list,in_features,drop_out_rate=None,is_head=False,num_classes=None):
        layers = []
        
        for i in range(len(layer_list)):
            
            out_features = layer_list[i]
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features
        
        if not is_head:
            layers.append(nn.Dropout(drop_out_rate))
            return nn.Sequential(*layers), out_features
        
        else:
            layers.append(nn.Linear(in_features, num_classes))
            return nn.Sequential(*layers)
    

def create_prameter(plans_path):
    
    with open(plans_path,"rb") as fr:
        plans = pickle.load(fr)['plans']
    
    stage_plans = plans['plans_per_stage'][0]
    parmeter_dict = {"input_channels" : plans['num_modalities'],
                    'base_num_features' : plans['base_num_features'],
                    'num_classes' : 2,
                    'num_pool' : len(stage_plans['pool_op_kernel_sizes']),
                    'num_conv_per_stage' : plans['conv_per_stage'],
                    'feat_map_mul_on_downscale' : 2, 
                    'conv_op': nn.Conv3d,
                    'norm_op': nn.InstanceNorm3d, 
                    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                    'dropout_op': nn.Dropout3d, 
                    'dropout_op_kwargs':{'p': 0, 'inplace': True},
                    'nonlin': nn.LeakyReLU,  
                    'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
                    'deep_supervision' : False, 
                    'dropout_in_localization': False,
                    'final_nonlin' : lambda x: x, 
                    'weightInitializer' : InitWeights_He(1e-2),
                    'pool_op_kernel_sizes' : stage_plans['pool_op_kernel_sizes'],
                    'conv_kernel_sizes' : stage_plans['conv_kernel_sizes'],
                    'upscale_logits' : False, 
                    'convolutional_pooling' : True, 
                    'convolutional_upsampling' : True}

    return parmeter_dict

class nnUnetBackbone(Generic_UNet):
    """Training with image only"""

    def __init__(self, plans_path, seg_weight=None, weight=None):
        
        parameter = create_prameter(plans_path)
        super().__init__(**parameter)

        if seg_weight :
            self.load_state_dict(torch.load(seg_weight)['state_dict'],strict=False)
        
        elif weight:
            checkpoint = torch.load(weight)
            checkpoint['state_dict'] = {k.replace('model.',''):v for k,v in checkpoint['state_dict'].items()}
            self.load_state_dict(torch.load(weight)['state_dict'],strict=False)
    
    def forward(self,x):

        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        x = self.conv_blocks_context[-1](x)
        img_em = F.adaptive_avg_pool3d(x, (1, 1,1)).view(x.shape[0],-1)
   
        return img_em
    

class nnUnetMultiModalFeatureConcat(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,weight)

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(320+clinical_feature_len, int((320+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((320+clinical_feature_len)/2), num_classes)
            )
        
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class nnUnetMultiModalProbConcat(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,weight)
        self.fc = nn.Linear(320, num_classes)

        if head == 'linear':
            self.head = nn.Linear(num_classes+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(num_classes+clinical_feature_len, int((num_classes+clinical_feature_len)/2)),
                nn.ReLU(inplace=True),
                nn.Linear(int((num_classes+clinical_feature_len)/2), num_classes)
            )
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = self.fc(x)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class nnUnetMultiModalEnsenble(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,weight)
        self.fc = nn.Linear(320, num_classes)

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
        img = self.fc(img)
        clinical = self.head(clinical)
        out = (img + clinical)/2
        return out
    

class nnUnetMultiModalPaper(nn.Module):
    def __init__(self, clinical_feature_len,
                 plans_path, 
                 hidden_dims_1,
                 hidden_dims_2,
                 hidden_dims_3,
                 drop_out_rate_1,
                 drop_out_rate_2,
                 drop_out_rate_3,
                 num_classes=2,weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,weight)

        
        self.fc1 = nn.Sequential(nn.Linear(320, hidden_dims_1[0]), 
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
    
    

class nnUnetMultiModalPaperGL(nn.Module):
    def __init__(self, clinical_feature_len,
                 plans_path, 
                 fc1_list,
                 fc2_list,
                 head_list,
                 drop_out_rate_1,
                 drop_out_rate_2,
                 num_classes=2,weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,weight)
        self.fc1,output_feature1 = create_layers(fc1_list,320,drop_out_rate_1)
        self.fc2,output_feature2 = create_layers(fc2_list,clinical_feature_len,drop_out_rate_2)
        self.head =  create_layers(head_list,output_feature1+output_feature2,is_head=True,num_classes=num_classes)
    
    def forward(self, img, clinical):
        img = self.backbone(img)
        img = img.flatten(start_dim=1)
        img = self.fc1(img)
        clinical =  self.fc2(clinical)
        out = self.head(torch.cat([img, clinical], dim=1))
        return out
    


class nnUnetMultiModalFeatureConcatTest(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,seg_weight,weight)
        self.clinical_backbone = nn.Sequential(nn.Linear(clinical_feature_len, int(clinical_feature_len/2)),nn.LeakyReLU())
        self.backbone_mlp =nn.Sequential(nn.Linear(320, 160),nn.LeakyReLU(),nn.Linear(160, 80),nn.LeakyReLU(),nn.Linear(80, 20),nn.LeakyReLU())

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(20+int(clinical_feature_len/2), int((20+int(clinical_feature_len/2))/2)),
                nn.LeakyReLU(inplace=True),
                nn.Linear(int((20+int(clinical_feature_len/2))/2), num_classes)
            )
        
        self.img_head = nn.Linear(20, num_classes)
        self.clinical_head = nn.Linear(int(clinical_feature_len/2), num_classes)
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        x_mlp = self.backbone_mlp(x)
        c_x = self.clinical_backbone(clinical)
        img_out = self.img_head(x_mlp)
        clinical_out = self.clinical_head(c_x)
        concat_x = torch.cat([x_mlp, c_x], dim=1)
        out = self.head(concat_x)
        return img_out, clinical_out, out
    
class nnUnetMultiModalFeatureConcatTest2(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone =nnUnetBackbone(plans_path,seg_weight,weight)
        self.clinical_backbone = nn.Sequential(nn.Linear(clinical_feature_len, int(clinical_feature_len/2)),nn.LeakyReLU())

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(320+int(clinical_feature_len/2), int((320+int(clinical_feature_len/2))/2)),
                nn.LeakyReLU(inplace=True),
                nn.Linear(int((320+int(clinical_feature_len/2))/2), num_classes)
            )
        
        self.img_head = nn.Linear(320, num_classes)
        self.clinical_head = nn.Linear(int(clinical_feature_len/2), num_classes)
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        c_x = self.clinical_backbone(clinical)
        img_out = self.img_head(x)
        clinical_out = self.clinical_head(c_x)
        concat_x = torch.cat([x, c_x], dim=1)
        out = self.head(concat_x)
        return img_out, clinical_out, out
    


class nnUnetMultiModalFeatureConcatTest3(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone =nnUnetBackbone(plans_path,seg_weight,weight)
        self.clinical_backbone = nn.Sequential(nn.Linear(clinical_feature_len, int(clinical_feature_len/2)),nn.LeakyReLU())

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(320+int(clinical_feature_len/2), int((320+int(clinical_feature_len/2))/2)),
                nn.LeakyReLU(inplace=True),
                nn.Linear(int((320+int(clinical_feature_len/2))/2), num_classes)
            )
        
        self.img_head = nn.Linear(320, num_classes)
        self.clinical_head = nn.Linear(int(clinical_feature_len/2), num_classes)
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        c_x = self.clinical_backbone(clinical)
        img_out = self.img_head(x)
        clinical_out = self.clinical_head(c_x)
        concat_x = torch.cat([x, c_x], dim=1)
        concat_x = nn.Dropout(0.5)(concat_x)
        out = self.head(concat_x)
        mean_out = torch.mean(torch.stack([img_out, clinical_out, out]), dim=0)
        return img_out, clinical_out, out ,mean_out
    
class nnUnetMultiModalFeatureConcatTest4(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone =nnUnetBackbone(plans_path,seg_weight,weight)
        self.head = SoftOrdering1DCNN(320+clinical_feature_len,num_classes,sign_size=320+clinical_feature_len)
        self.img_head = nn.Linear(320, num_classes)
        self.clinical_head = nn.Linear(clinical_feature_len, num_classes)

    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        c_out = self.clinical_head(clinical)
        img_out = self.img_head(x)
        concat_x = torch.cat([x, clinical], dim=1)
        out = self.head(concat_x)
        mean_out = torch.mean(torch.stack([img_out, c_out, out]), dim=0)
        return img_out, c_out, out ,mean_out

class nnUnetMultiModalFeatureConcat1DCNN(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone = nnUnetBackbone(plans_path,seg_weight,weight)
        self.head = SoftOrdering1DCNN(320+clinical_feature_len,num_classes,sign_size=320+clinical_feature_len)
        self.img_head = nn.Linear(320,num_classes)
        
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        x = torch.cat([x, clinical], dim=1)
        x = self.head(x)
        return x
    

class nnUnetMultiModalFeatureConcatTest5(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone =nnUnetBackbone(plans_path,seg_weight,weight)
        self.clinical_backbone = nn.Sequential(nn.Linear(clinical_feature_len, int(clinical_feature_len/2)),nn.LeakyReLU())

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(2*int(clinical_feature_len/2), int((2*int(clinical_feature_len/2))/2)),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2*((40+int(clinical_feature_len/2))/2), num_classes)
            )
        
        self.img_head = nn.Linear(40, num_classes)
        self.clinical_head = nn.Linear(int(clinical_feature_len/2), num_classes)
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        
        x = nn.Linear(320, 160)(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = nn.Linear(160, 80)(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = nn.Linear(80, int(clinical.shape[1]/2))(x)
        x = nn.LeakyReLU(inplace=True)(x)
        
        c_x = self.clinical_backbone(clinical)
        img_out = self.img_head(x)
        clinical_out = self.clinical_head(c_x)
        x = x + c_x
        out = self.head(x)
        mean_out = torch.mean(torch.stack([img_out, clinical_out, out]), dim=0)
        return img_out, clinical_out, out ,mean_out
    

class nnUnetMultiModalFeatureConcatTest6(nn.Module):
    def __init__(self, clinical_feature_len,plans_path,head='linear', num_classes=2,weight=None,seg_weight=None) -> None:
        super().__init__()
        
        self.backbone =nnUnetBackbone(plans_path,seg_weight,weight)
        self.clinical_backbone = nn.Sequential(nn.Linear(clinical_feature_len, int(clinical_feature_len/2)),nn.LeakyReLU())

        if head == 'linear':
            self.head = nn.Linear(320+clinical_feature_len, num_classes)
    
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(2*int(clinical_feature_len/2), 2*int((2*int(clinical_feature_len/2))/2)),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2*int((2*int(clinical_feature_len/2))/2), num_classes)
            )
        
        self.img_head = nn.Linear(40, num_classes)
        self.clinical_head = nn.Linear(int(clinical_feature_len/2), num_classes)
    
    def forward(self, img, clinical):
        x = self.backbone(img)
        x = x.flatten(start_dim=1)
        
        x = nn.Linear(320, 160)(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = nn.Linear(160, 80)(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = nn.Linear(80, int(clinical.shape[1]/2))(x)
        x = nn.LeakyReLU(inplace=True)(x)
        
        c_x = self.clinical_backbone(clinical)
        img_out = self.img_head(x)
        clinical_out = self.clinical_head(c_x)
        x = x * c_x
        out = self.head(x)
        mean_out = torch.mean(torch.stack([img_out, clinical_out, out]), dim=0)
        return img_out, clinical_out, out ,mean_out
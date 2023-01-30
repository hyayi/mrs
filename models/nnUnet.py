import torch 
import pickle
import torch.nn as nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
import torch.nn.functional as F

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

    def __init__(self, plans_path, num_classes, weight=None):
        
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
        return img_em, out
    
import pandas as pd 
from sklearn.model_selection import StratifiedKFold
import pickle
import torch.nn as nn
from nnunet.network_architecture.initialization import InitWeights_He
import random 
import os 
import numpy as np
import torch 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def data_load(train_path:str,  
              fold = 0) :

    total = pd.read_csv(train_path)
    fold = f"fold{fold}"
    train = total[total[fold] == "train"]
    val = total[total[fold] == "val"]
    
    print(fold)    
    print(f"train counte:{len(train)}, train 0 count : {train['label'].value_counts()[0]}, train 1 count : {train['label'].value_counts()[1]}")
    print(f"val counte:{len(val)}, val 0 count : {val['label'].value_counts()[0]}, val 1 count : {val['label'].value_counts()[1]}")
    
    return train, val

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
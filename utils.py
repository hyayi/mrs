import numpy as np
import pandas as pd 
import torch 
import random
import os
from sklearn.model_selection import StratifiedKFold

def return_top_k_slice(mask_np,top_k):
    return np.sort(np.argpartition(np.sum(mask_np,axis=(0,1)),(-top_k))[-top_k:])

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def data_load(train_path, 
              val_path, 
              test_path,
              fold = None,
              fold_num = None):
    
    train = pd.read_csv(train_path)
    val =pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    
    if fold_num is not None :
        train_fold = pd.concat([train,val],axis=0)
        kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
        
        for i, (train_index, val_index) in enumerate(kf.split(train_fold,train_fold['label'])):
            if i == fold:
                train = train_fold.iloc[train_index].reset_index(drop=True)
                val = train_fold.iloc[val_index].reset_index(drop=True)
                break
    
    print(fold)    

    print(f"train counte:{len(train)}, train 0 count : {train['label'].value_counts()[0]}, train 1 count : {train['label'].value_counts()[1]}")
    print(f"val counte:{len(val)}, val 0 count : {val['label'].value_counts()[0]}, val 1 count : {val['label'].value_counts()[1]}")
    print(f"test counte:{len(test)}, test 0 count : {test['label'].value_counts()[0]}, test 1 count : {test['label'].value_counts()[1]}")
    
    
    return train, val, test
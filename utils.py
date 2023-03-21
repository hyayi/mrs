import pandas as pd 
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import os 
import random 
import numpy as np

def get_values(value):
    return value.values.reshape(-1, 1)

def clinical_data_preprocessing(train, 
                                val,
                                test,
                                scaler_name,
                                categorical_cols,
                                numerical_cols,
                                ignore_cols):
    
    colums = train.keys()
    for col in colums :

        if col in ignore_cols:
            continue

        elif col in numerical_cols:
            scaler = getattr(sklearn.preprocessing, scaler_name)()
            train[col] = scaler.fit_transform(get_values(train[col]))
            val[col] = scaler.transform(get_values(val[col]))
            test[col] = scaler.transform(get_values(test[col]))

        elif col in categorical_cols:

            
            one_hot = OneHotEncoder(sparse=False)
            train_onehot = pd.DataFrame(one_hot.fit_transform(get_values(train[col].astype('int'))), columns=[f"{col}_{value}" for value in np.sort(train[col].unique())])
            val_onehot = pd.DataFrame(one_hot.transform(get_values(val[col].astype('int'))), columns=[f"{col}_{value}" for value in np.sort(val[col].unique())])
            test_onehot = pd.DataFrame(one_hot.transform(get_values(test[col].astype('int'))), columns=[f"{col}_{value}" for value in np.sort(test[col].unique())])
            
            train.drop(col,axis=1,inplace=True)
            val.drop(col,axis=1,inplace=True)
            test.drop(col,axis=1, inplace=True)

            train = pd.concat([train,train_onehot], axis=1)
            val = pd.concat([val,val_onehot], axis=1)
            test = pd.concat([test,test_onehot], axis=1)

        else:
            continue
        
    return train, val, test
    
def data_load(train_path, 
              val_path, 
              test_path,
              scaler_name = None,
              categorical_cols = None, 
              numerical_cols = None, 
              ignore_cols = None,
              fold = None,
              fold_num = None):

    train = pd.read_csv(train_path)
    val =pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    target = ignore_cols + categorical_cols+numerical_cols 
    
    if fold_num is not None :
        train_fold = pd.concat([train,val],axis=0)
        kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=42)
        for i, (train_index, val_index) in enumerate(kf.split(train_fold,train_fold['label'])):
            if i == fold:
                train = train_fold.iloc[train_index].reset_index(drop=True)
                val = train_fold.iloc[val_index].reset_index(drop=True)
                train = train[target]
                val = val[target]
                break
    train, val, test = clinical_data_preprocessing(train[target], val[target], test[target], scaler_name, categorical_cols, numerical_cols, ignore_cols)
    
    print(fold)    
    print(f"train counte:{len(train)}, train 0 count : {train['label'].value_counts()[0]}, train 1 count : {train['label'].value_counts()[1]}")
    print(f"val counte:{len(val)}, val 0 count : {val['label'].value_counts()[0]}, val 1 count : {val['label'].value_counts()[1]}")
    print(f"test counte:{len(test)}, test 0 count : {test['label'].value_counts()[0]}, test 1 count : {test['label'].value_counts()[1]}")
    
    
    return train, val, test

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

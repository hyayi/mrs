import pandas as pd 
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
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
              ignore_cols = None,):
    
    drop_col = ['wbc','hb', 'plt', 'tc', 'tg', 'hdl', 'ldl', 'bun', 'cr', 'fbs', 'sbp', 'dbp','toast']
    # categorical_cols = ['sex','toast','hx_str','hx_htn','hx_smoke','hx_af','tx_throm']
    # numerical_cols = ['age','bmi','ini_nih','pre_mrs','wbc','hb', 'plt', 'tc', 'tg', 'hdl', 'ldl', 'bun', 'cr', 'fbs', 'sbp', 'dbp']
    # ignore_cols = ['image','label','mask_path']

    train = pd.read_csv(train_path)
    val =pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
    
    train.drop(drop_col, axis=1, inplace=True)
    val.drop(drop_col, axis=1, inplace=True)
    test.drop(drop_col, axis=1, inplace=True)


    train, val, test = clinical_data_preprocessing(train, val, test, scaler_name, categorical_cols, numerical_cols, ignore_cols)
        
    print(f"train counte:{len(train)}, train 0 count : {train['label'].value_counts()[0]}, train 1 count : {train['label'].value_counts()[1]}")
    print(f"val counte:{len(val)}, val 0 count : {val['label'].value_counts()[0]}, val 1 count : {val['label'].value_counts()[1]}")
    print(f"test counte:{len(test)}, test 0 count : {test['label'].value_counts()[0]}, test 1 count : {test['label'].value_counts()[1]}")
    
    
    return train, val, test
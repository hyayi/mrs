import pandas as pd 


def data_load(train_path, 
              val_path, 
              test_path):
    
    train = pd.read_csv(train_path)
    val =pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)
        
    print(f"train counte:{len(train)}, train 0 count : {train['label'].value_counts()[0]}, train 1 count : {train['label'].value_counts()[1]}")
    print(f"val counte:{len(val)}, val 0 count : {val['label'].value_counts()[0]}, val 1 count : {val['label'].value_counts()[1]}")
    print(f"test counte:{len(test)}, test 0 count : {test['label'].value_counts()[0]}, test 1 count : {test['label'].value_counts()[1]}")
    
    
    return train, val, test

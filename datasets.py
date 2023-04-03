import torch

class MRSMultiDataset(torch.utils.data.Dataset):
    def __init__(self, data_df,data_dir, transforms):
        self.data_df = data_df
        self.transforms = transforms
        self.data_dir = data_dir
        
    def prerpocessing(self,df):
        clinical_data_feature = df.iloc[:,3:].values
        return torch.as_tensor(clinical_data_feature, dtype =torch.float)
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = torch.as_tensor(self.data_df['label'][index])

        image_path = f"{self.data_dir}/{self.data_df['image'][index]}.nii.gz"
        image_dict =  {'img' : image_path}
        image_data = self.transforms(image_dict)
        
        clinical_data = self.prerpocessing(self.data_df)[index]
        
        return image_data['img'],clinical_data, label
    
class MRSMultiInferDataset(torch.utils.data.Dataset):
    def __init__(self, data_df,data_dir, transforms):
        self.data_df = data_df
        self.transforms = transforms
        self.data_dir = data_dir
        
    def prerpocessing(self,df):
        clinical_data_feature = df.iloc[:,3:].values
        return torch.as_tensor(clinical_data_feature, dtype =torch.float)
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = torch.as_tensor(self.data_df['label'][index])

        image_path = f"{self.data_dir}/{self.data_df['image'][index]}.nii.gz"
        image_dict =  {'img' : image_path}
        image_data = self.transforms(image_dict)
        
        clinical_data = self.prerpocessing(self.data_df)[index]
        
        return image_data['img'],clinical_data, label, self.data_df['image'][index]
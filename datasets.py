import torch

class MRSImaOlnyDataset(torch.utils.data.Dataset):
    def __init__(self, data_df,data_dir, transforms):
        self.data_df = data_df
        self.transforms = transforms
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = torch.as_tensor(self.data_df['label'][index])
        image_path = f"{self.data_dir}/{self.data_df['image'][index]}_0000.nii.gz"
        image_dict =  {'img' : image_path}
        image_data = self.transforms(image_dict)
        img = image_data['img']
        img = torch.transpose(image_data['img'], 1, 3)
        img = torch.transpose(img, 2, 3)
        print(img.shape)
        return img, label


class MRSImaOlnyDatasetInfer(torch.utils.data.Dataset):
    def __init__(self, data_df,data_dir, transforms):
        self.data_df = data_df
        self.transforms = transforms
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = torch.as_tensor(self.data_df['label'][index])
        
        image_path = f"{self.data_dir}/{self.data_df['image'][index]}.nii.gz"
        image_dict =  {'img' : image_path}
        image_data = self.transforms(image_dict)
    
        return image_data['img'], label, self.data_df['image'][index]

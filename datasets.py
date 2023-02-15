import numpy as np
import torch
import nibabel as nib
from utils import return_top_k_slice
import cv2

class MRSDataset2D(torch.utils.data.Dataset):
    def __init__(self, data_df,data_dir, transforms):
        self.data_df = data_df
        self.transforms = transforms
        self.data_dir = data_dir
         
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        
        label = torch.as_tensor(self.data_df['label'][index])
        image_path = f"{self.data_dir}/{self.data_df['image'][index]}.nii.gz"
        mask_path = f"{self.data_dir}/{self.data_df['mask'][index]}.nii.gz"

        img = nib.load(image_path)
        mask = nib.load(mask_path)
    
        ##orientation 수정 
        img = nib.as_closest_canonical(img)
        mask = nib.as_closest_canonical(mask)
        
        img = img.get_fdata()
        mask = mask.get_fdata()
        
        top_3 = return_top_k_slice(mask,3)
        
        img = img[:,:,top_3]
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return img, label


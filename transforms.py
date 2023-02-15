import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(data):
    if data == 'train':
        return A.Compose([
            A.Normalize(max_pixel_value=255),
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()])
        
    elif data == 'test':
        return A.Compose([
            A.Normalize(max_pixel_value=255),
            A.Resize(height=224, width=224),
            ToTensorV2()])
        
        

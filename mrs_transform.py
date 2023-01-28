from monai.transforms import LoadImaged, AddChanneld, Orientationd, ScaleIntensityd, RandRotated,Resized, RandShiftIntensityd, EnsureTyped, Compose,Spacingd,RandAxisFlipd, RandAdjustContrastd,RandBiasFieldd
import numpy as np

def get_trasforms(pixdim,axcodes,spatial_size) :

    train_transforms = Compose(
        [
            LoadImaged(keys="img"),
            AddChanneld(keys="img"),
            Spacingd(keys='img',pixdim=pixdim),
            Orientationd(keys='img',axcodes=axcodes),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=spatial_size),
            RandRotated(keys=["img"], range_x=np.pi / 12, prob=0.3, keep_size=True),
            RandAxisFlipd(keys=["img"],prob=0.3),
            EnsureTyped(keys=["img"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys="img"),
            AddChanneld(keys="img"),
            Spacingd(keys='img',pixdim=pixdim),
            Orientationd(keys='img',axcodes=axcodes),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=spatial_size),
            EnsureTyped(keys=["img"]),
        ]
    ) 
    return train_transforms, val_transforms
        
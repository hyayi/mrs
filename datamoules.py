from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets
import transforms
from monai.transforms import Compose
from utils import data_load


class BrainDataModule(pl.LightningDataModule):
    

    def __init__(self,data_config,fold=None,fold_num=None):
        super().__init__()
        
        self.config =data_config
        self.dataset = getattr(datasets,self.config['dataset']['name'])
        self.train,self.val, self.test = data_load(fold=fold,fold_num=fold_num,**self.config['data_load_params'])
        
        self.train_transforms = Compose([getattr(transforms,name)(**params) for name, params in self.config['transforms']['train'].items()])
        self.val_transforms = Compose([getattr(transforms,name)(**params) for name, params in self.config['transforms']['val'].items()])
        
        self.class_weights = 1 - self.train['label'].value_counts(normalize=True).values
        
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.dataset(data_df=self.train, transforms= self.train_transforms, data_dir=self.config['dataset']['params']['data_dir'])
            self.val_ds = self.dataset(data_df=self.val, transforms= self.val_transforms, data_dir=self.config['dataset']['params']['data_dir'])

        if stage == 'test' or stage is None:
            self.test_ds = self.dataset(data_df=self.test, transforms= self.val_transforms, data_dir=self.config['dataset']['params']['data_dir'])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'],shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'],shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'])
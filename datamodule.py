from torch.utils.data import DataLoader
import pytorch_lightning as pl
import datasets
from transforms import get_transforms
from utils import data_load
import pandas as pd


class BrainDataModule(pl.LightningDataModule):
    

    def __init__(self,data_config):
        super().__init__()
        
        self.config =data_config
        self.dataset = getattr(datasets,self.config['dataset']['name'])
        self.train,self.val, self.test = data_load(**self.config['data_load_params'])
        self.class_weights = 1 - self.train['label'].value_counts(normalize=True).values
        
    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_ds = self.dataset(data_df=self.train, transforms= get_transforms('train'), data_dir=self.config['dataset']['params']['data_dir'])
            self.val_ds = self.dataset(data_df=self.val, transforms= get_transforms('test'), data_dir=self.config['dataset']['params']['data_dir'])

        if stage == 'test' or stage is None:
            self.test_ds = self.dataset(data_df=self.test, transforms= get_transforms('test'), data_dir=self.config['dataset']['params']['data_dir'])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'],shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'],shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config['dataloader']['params']['batch_size'],num_workers=self.config['dataloader']['params']['num_workers'], pin_memory=self.config['dataloader']['params']['pin_memory'])
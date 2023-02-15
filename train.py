import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pytorch_lightning as pl
import lighthining_model
import  datamoules 
import argparse
import yaml
import torch
import warnings

import numpy as np
import random
import torch.multiprocessing
from pytorch_lightning.loggers import WandbLogger

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
    
def train(args):
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dm = getattr(datamoules, config['datamodule']['name'])(data_config=config['datamodule'])
    class_weights = data_dm.class_weights
    model = getattr(lighthining_model, config['lighthining_model']['name'])(model_config=config['lighthining_model'],class_weights=class_weights)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{args.save_path}/", save_top_k=1, monitor="val_auc",filename=f'{args.model_name}'+'-{epoch:02d}-{val_auc:.3f}',mode='max')
    callbacks = [checkpoint_callback]
    wandb_logger = WandbLogger(project="BrainMrs", name=args.model_name, save_dir=f"{args.save_path}")
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,callbacks=callbacks,strategy=args.strategy,logger=wandb_logger,**config['trainer'])
    
    trainer.fit(model,data_dm)
    data_dm.setup()
    trainer.validate(model, ckpt_path=checkpoint_callback.best_model_path, dataloaders = data_dm.test_dataloader())
    

if __name__=="__main__" :
    
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default='config\experiment2.yaml')
    parser.add_argument("--model_name", type=str, default='test')
    parser.add_argument("--accelerator", type=str, default= 'gpu')
    parser.add_argument("--save_path", type=str, default= './')
    parser.add_argument("--devices", type=int, default= 1)
    parser.add_argument("--strategy", type=str, default= None)
    
    args = parser.parse_args()
    
    train(args)
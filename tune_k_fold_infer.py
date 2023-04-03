import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pytorch_lightning as pl
import lighthining_model
import  datamoules 
import argparse
import yaml
import torch
import warnings

import torch.multiprocessing
from pytorch_lightning.loggers import WandbLogger
from utils import seed_everything
import pandas as pd 
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pickle
from lighthining_model import MRSClassficationMultiModal

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')
seed_everything(42)


def infer(args):

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    test_result = {}
    val_result = {}
    train_result = {}
    
    for fold in range(args.fold_num):
        data_dm = getattr(datamoules, config['datamodule']['name'])(data_config=config['datamodule'],fold=fold, fold_num=args.fold_num)
        
        model = MRSClassficationMultiModal.load_from_checkpoint(args.model_path)
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices)
        
        train_predictions = trainer.predict(model, dataloaders = data_dm.train_infer_dataloader(), ckpt_path=args.model_path_list[fold])
        val_predictions = trainer.predict(model, dataloaders = data_dm.val_dataloader(), ckpt_path=args.model_path_list[fold])
        test_predictions = trainer.predict(model, dataloaders = data_dm.test_dataloader(),ckpt_path=args.model_path_list[fold])
        
        train_result[f'{fold}'] = train_predictions
        test_result[f'{fold}'] = test_predictions
        val_result[f'{fold}'] = val_predictions
    
    with open(f'{args.save_path}/train_result.pkl', 'wb') as f:
        pickle.dump(train_result, f)
        
    with open(f'{args.save_path}/test_result.pkl', 'wb') as f:
        pickle.dump(test_result, f)
        
    with open(f'{args.save_path}/val_result.pkl', 'wb') as f:
        pickle.dump(val_result, f)
        

if __name__=="__main__" :
    
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default='config\mc3_18_multi_paper_tune.yaml')
    parser.add_argument("--model_name", type=str, default='muti_paper_tune')
    parser.add_argument("--accelerator", type=str, default= 'gpu')
    parser.add_argument("--save_path", type=str, default= './test/')
    parser.add_argument("--devices", type=int, default= 1)
    parser.add_argument("--strategy", type=str, default= None)
    parser.add_argument("--model_path", action='append', dest='--model_path_list')
    parser.add_argument("--fold_num", type=int, default= 5)
    
    args = parser.parse_args()
    
    infer(args)
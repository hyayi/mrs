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

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')
seed_everything(42)
    
def train(args):
    
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    test_result = {}
    val_result = {}
    
    for fold in range(args.fold_num):
        data_dm = getattr(datamoules, config['datamodule']['name'])(data_config=config['datamodule'],fold=fold, fold_num=args.fold_num)
        class_weights = data_dm.class_weights
        
        model = getattr(lighthining_model, config['lighthining_model']['name'])(model_config=config['lighthining_model'],class_weights=class_weights)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{args.save_path}/", save_top_k=1, monitor="val_auc",filename=f'{args.model_name}-{fold}'+ '-{epoch:02d}-{val_auc:.3f}',mode='max')
        callbacks = [checkpoint_callback]
        
        wandb_logger = WandbLogger(project=args.model_name, name=f"{fold}", save_dir=f"{args.save_path}")
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,callbacks=callbacks,strategy=args.strategy,logger=wandb_logger,**config['trainer'])
        
        trainer.fit(model,data_dm)
        
        print(fold)
        val_metric_list = trainer.validate(model,datamodule = data_dm, ckpt_path=checkpoint_callback.best_model_path)
        test_metric_list = trainer.test(model,datamodule = data_dm, ckpt_path=checkpoint_callback.best_model_path)
        
        test_result[f'{fold}'] = test_metric_list[0]
        val_result[f'{fold}'] = val_metric_list[0]
    
    total_result = pd.concat([pd.DataFrame(val_result).T,pd.DataFrame(test_result).T], axis = 1)
    total_result.to_csv(f'{args.save_path}/{args.model_name}_total_result.csv')
    
    print(f'mean_val_score : {total_result["val_auc"].mean()} , std_val_score : {total_result["val_auc"].std()}')
    print(f'mean_test_score : {total_result["test_auc"].mean()} , std_test_score : {total_result["test_auc"].std()}')
    
if __name__=="__main__" :
    
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default='config\SimplyMlp.yaml')
    parser.add_argument("--model_name", type=str, default='simpy_mlp')
    parser.add_argument("--accelerator", type=str, default= 'gpu')
    parser.add_argument("--save_path", type=str, default= './')
    parser.add_argument("--devices", type=int, default= 1)
    parser.add_argument("--strategy", type=str, default= None)
    parser.add_argument("--fold_num", type=int, default= 5)
    
    args = parser.parse_args()
    
    train(args)
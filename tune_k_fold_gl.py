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

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings(action='ignore')
seed_everything(42)


def objective(trial):

    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    config['lighthining_model']['optimizer']['lr'] = trial.suggest_float("lr", 0.00001, 0.001)
    config['lighthining_model']['optimizer']['gamma'] = trial.suggest_float("gamma", 0.1, 0.9)
    
    config['lighthining_model']['model']['params']['fc1_list'] = [trial.suggest_int("1_units_l{}".format(i), 10, 320) for i in range(trial.suggest_int("n_layers", 1, 10))]
    config['lighthining_model']['model']['params']['fc2_list'] = [trial.suggest_int("2_units_l{}".format(i), 10, 320) for i in range(trial.suggest_int("n_layers", 1, 10))]
    config['lighthining_model']['model']['params']['head_list'] = [trial.suggest_int("head_units_l{}".format(i), 10, 320) for i in range(trial.suggest_int("n_layers", 1, 10))]
    
    config['lighthining_model']['model']['params']['drop_out_rate_1'] = trial.suggest_float("1_drop_out_rate", 0.1, 0.7)
    config['lighthining_model']['model']['params']['drop_out_rate_2'] = trial.suggest_float("2_drop_out_rate", 0.1, 0.7)

    
    test_result = {}
    val_result = {}
    train_result = {}
    
    for fold in range(args.fold_num):
        data_dm = getattr(datamoules, config['datamodule']['name'])(data_config=config['datamodule'],fold=fold, fold_num=args.fold_num)
        class_weights = data_dm.class_weights
        
        config['lighthining_model']['model']['params']['clinical_feature_len'] = data_dm.clincal_feature_len
        model = getattr(lighthining_model, config['lighthining_model']['name'])(model_config=config['lighthining_model'],class_weights=class_weights)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=f"{args.save_path}/", save_top_k=1, monitor="val_auc",filename=f'{args.model_name}-{trial.number}-{fold}'+ '-{epoch:02d}-{val_auc:.3f}',mode='max')
        callbacks = [checkpoint_callback,PyTorchLightningPruningCallback(trial, monitor="val_auc")]
        
        wandb_logger = WandbLogger(project=args.model_name, name=f"{fold}", save_dir=f"{args.save_path}")
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,callbacks=callbacks,strategy=args.strategy,logger = wandb_logger, **config['trainer'])
        
        trainer.fit(model,data_dm)
        
        print(fold)
        train_metric_list = trainer.test(model,dataloaders = data_dm.train_dataloader(), ckpt_path=checkpoint_callback.best_model_path)
        val_metric_list = trainer.test(model,dataloaders = data_dm.val_dataloader(), ckpt_path=checkpoint_callback.best_model_path)
        test_metric_list = trainer.test(model,dataloaders = data_dm.test_dataloader(), ckpt_path=checkpoint_callback.best_model_path)
    
        train_result[f'{fold}'] = train_metric_list[0]
        test_result[f'{fold}'] = test_metric_list[0]
        val_result[f'{fold}'] = val_metric_list[0]
    
    train_result_df = pd.DataFrame(train_result).T
    val_result_df = pd.DataFrame(val_result).T
    test_result_df = pd.DataFrame(test_result).T
    
    train_result_df.columns = [f'train_{col}' for col in train_result_df.columns]
    val_result_df.columns = [f'val_{col}' for col in val_result_df.columns]
    test_result_df.columns = [f'test_{col}' for col in test_result_df.columns]
    
    total_result = pd.concat([train_result_df,val_result_df,test_result_df], axis = 1)
    total_result.to_csv(f'{args.save_path}/{args.model_name}_{trial.number}_total_result.csv')
    
    print(f'mean_val_score : {total_result["val_auc"].mean()} , std_val_score : {total_result["val_auc"].std()}')
    print(f'mean_test_score : {total_result["test_auc"].mean()} , std_test_score : {total_result["test_auc"].std()}')
    
    return total_result["val_auc"].mean()

if __name__=="__main__" :
    
    print('start')
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default='config/nnunet/paper_list/nnunet_multi_paper_tune_all_gl_local.yaml')
    parser.add_argument("--model_name", type=str, default='muti_paper_tune')
    parser.add_argument("--accelerator", type=str, default= 'gpu')
    parser.add_argument("--save_path", type=str, default= './test/')
    parser.add_argument("--devices", type=int, default= 1)
    parser.add_argument("--strategy", type=str, default= None)
    parser.add_argument("--fold_num", type=int, default= 5)
    parser.add_argument("--n_trials", type=int, default= 5)
    
    global args
    args = parser.parse_args()
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    study.trials_dataframe().to_csv(f'{args.save_path}/{args.model_name}_optuna_result.csv')

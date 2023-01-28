#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
import sys
from utils import *
from tqdm import tqdm
from dataset1 import Lung3D_ccii_patient_supcon, MRSupcon
from torch.utils.data import DataLoader
import torch.nn.functional as F
from visualize import Visualizer
from torchnet import meter
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from models.ResNet import SupConResNet

import torch.backends.cudnn as cudnn
import random
import math
from mrs_transform import get_trasforms
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional import auroc
import wandb

## torch version 출력
print("torch = {}".format(torch.__version__))  

## 시간 출력
IMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

## argpaser 
parser = argparse.ArgumentParser()
parser.add_argument('--visname', '-vis', default='2d', help='visname') ## 시각화 관련 
parser.add_argument('--batch_size', '-bs', default=4, type=int, help='batch_size')
parser.add_argument('--lr', '-lr', default=1e-4, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n_cls', default=2, type=int, help='n_classes')
parser.add_argument('--distpre', '-pre', default=False, type=bool, help='use pretrained')
parser.add_argument('--data_dir', '-data', default='D:/study_d/project/brain/code/data/image', help='data_dir')
parser.add_argument('--train_data_path', '-train', default='D:/study_d/project/brain/code/data/annotation/train.csv', help='train_data_path')
parser.add_argument('--val_data_path', '-val', default='D:/study_d/project/brain/code/data/annotation/val.csv', help='val_data_path')
parser.add_argument('--test_data_path', '-test', default='D:/study_d/project/brain/code/data/annotation/test.csv', help='test_data_path')
parser.add_argument("--spatial_size",nargs='+',type=int, default=[48,256,256])
parser.add_argument("--pixdim",nargs='+',type=int,default=[1,1,5])
parser.add_argument("--axcodes",type=str,default='SPL')
parser.add_argument('--project_name', '-project', default='mrs_scop', help='project_name')


## checpoint 할려고 global변수로 best_f1을 선억했고
best_auc = 0
val_epoch = 1
save_epoch = 10


### 재현성을 위해 seed 설정 
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.deterministic = True

## paraser 
def parse_args():
    global args
    args = parser.parse_args()

### 학습 진행합에 따라 lr를 감소시킬려는 부분 
def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr

def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr
parse_args()

run = wandb.init(project=args.project_name)

cfg = {
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
}

wandb.config.update(cfg)

def main():
    print(torch.cuda.device_count())  ## gpu 갯수 출력 
    global best_auc
    global save_dir
    
    train_df,val_df,test_df = load_data(args.train_data_path,args.val_data_path,args.test_data_path) ## 데이터 로드
    train_transforms, val_transfomrs = get_trasforms(pixdim = tuple(args.pixdim),axcodes=args.axcodes,spatial_size=tuple(args.spatial_size))

    # vis = Visualizer(args.visname,port=9000) ## 시각화 툴
    # prepare the model
    target_model = SupConResNet(name='resnest50_3D', head='mlp', feat_dim=128, n_classes=args.n_classes) ## 모델 선언

    s1 = target_model.sigma1   ## adaptive joint training strategy에 사용하는 sigma1, sigma2
    s2 = target_model.sigma2

    # ccii-pre
    # ckpt = torch.load('/remote-home/junlinHou/2019-nCov/CC-CCII/3dlung/checkpoints/con/ccii3d2clfres50supcon/19.pkl')
    # state_dict = ckpt['net']
    # unParalled_state_dict = {}
    # for key in state_dict.keys():
    #     unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    # target_model.load_state_dict(unParalled_state_dict,True)

    target_model = nn.DataParallel(target_model)  ## muli gpu설정 
    target_model = target_model.cuda()
    
    # prepare data
    ## 데이터 셋 설정 
    train_data = MRSupcon(data_df=train_df, data_dir=args.data_dir,transforms=train_transforms,mode='train')   
    val_data = MRSupcon(data_df=val_df, data_dir=args.data_dir,transforms=val_transfomrs, mode='val')

    ## loaer 설정 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,pin_memory=True)

    ##loss 및 optimizer  설정 
    criterion = SupConLoss(temperature=0.1)
    criterion = criterion.cuda()
    criterion_clf = nn.CrossEntropyLoss()
    criterion_clf = criterion_clf.cuda()
    optimizer = torch.optim.Adam(target_model.parameters(), args.lr, weight_decay=1e-5)

    ## confusion metrais를 시각화 하기 위해 
    con_matx = meter.ConfusionMeter(args.n_classes)

    ## 저장 장소 
    save_dir = './checkpoints/con/'+ str(args.visname)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  

    # train the model

    # initial_epoch = checkpoint['epoch']
    initial_epoch = 0
    for epoch in range(initial_epoch, initial_epoch + args.epochs):
        target_model = target_model.train()
        con_matx.reset()
        total_loss1 = .0
        total_loss2 = .0
        total = .0
        correct = .0
        count = .0
        total_num = .0

        lr = args.lr
        
        ## 각 파라미터에 대해 lr을 설정
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        pred_list = []
        label_list = []
        
        pbar = tqdm(train_loader, ascii=True) ## loder를 tqdm으로 설정 
   
        for i, (imgs, labels, ID) in enumerate(pbar):
            imgs = torch.cat([imgs[0],imgs[1]],dim=0) #2*bsz,256,256  ## agementation한 이미지를 합쳐서 넣어주는 과정인거 같은데 아마 한개의 batch로 합칠려는 거 같음 
            #imgs = imgs.unsqueeze(1).float().cuda() #2*bsz,1,256,256 ## ct이미지라서 저렇게 한거 같긴한데 의문점은 3d를 사용했는데 왜 차원은 2차원으로 적어둔거지? 
            labels = labels.float().cuda()
            bsz = labels.shape[0]

            features, pred = target_model(imgs) #2*bsz,128 #2*bsz,n_class  모델에 이미지를 넣어서 feature contrativeloss용 와 pred #classficatoin를 얻어옴
            f1, f2 = torch.split(features, [bsz, bsz], dim=0) #bsz,128  agmentiont 것들을 나누기를 진행 
            features = torch.cat([f1.unsqueeze(1),f2.unsqueeze(1)],dim=1) #bsz,2,128 ## 이거는 loss를 들어과 봐야알겠네 
            loss_con = criterion(features,labels)

            pred1, pred2 = torch.split(pred, [bsz, bsz], dim=0) #bsz,n_classs
            pred1 = F.softmax(pred1)
            pred2 = F.softmax(pred2)
            con_matx.add(pred1.detach(),labels.detach())
            con_matx.add(pred2.detach(),labels.detach())
            _, predicted1 = pred1.max(1)
            _, predicted2 = pred2.max(1)
            loss_clf = 0.5*criterion_clf(pred1,labels.long())+0.5*criterion_clf(pred2,labels.long())  ## agemetation된 결과들을 합쳐서 loss를 구하는 부분 이군 

            pred_list.append(predicted1.cpu().detach())
            label_list.append(labels.cpu().detach())
            pred_list.append(predicted2.cpu().detach())
            label_list.append(labels.cpu().detach())

            loss = torch.exp(-s1)*loss_con+s1+torch.exp(-s2)*loss_clf+s2 ## adaptive loss 전략 
            # loss = loss_clf
            total_loss1 += loss_con.item()
            total_loss2 += loss_clf.item()
            total += 2 * bsz
            correct += predicted1.eq(labels.long()).sum().item()
            correct += predicted2.eq(labels.long()).sum().item()
            count += 1
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            pbar.set_description('loss: %.3f' % (total_loss2 / (i+1))+' acc: %.3f' % (correct/total))

        recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
        
        #vis.plot('loss1_con', total_loss1/count)
        #vis.plot('loss', total_loss2/count)
        #vis.log('epoch:{epoch},lr:{lr},loss1:{loss1},loss2:{loss2}'.format(epoch=epoch,lr=lr,loss1=total_loss1/count,loss2=total_loss2/count))
        wandb.log({'epoch': epoch, 'lr':lr,'train_loss1':total_loss1/count,'train_loss2':total_loss2/count, 'train_acc' :correct/total} ,step=epoch)
        if (epoch + 1) % val_epoch == 0:
            val1(target_model,val_loader,epoch)
            print(torch.exp(-s1).item(),torch.exp(-s2).item())
         

@torch.no_grad()
def val1(net, val_loader, epoch):
    global best_auc
    parse_args()
    net = net.eval()

    correct = .0
    total = .0
    con_matx = meter.ConfusionMeter(args.n_classes)
    pred_list = []
    label_list = []
  
    pbar = tqdm(val_loader, ascii=True)
    for i, (data, label,id) in enumerate(pbar):
        #data = data.unsqueeze(1)
        data = data.float().cuda()
        label = label.float().cuda()
        _, pred = net(data)
        pred = F.softmax(pred)
        _, predicted = pred.max(1)

        pred_list.append(predicted.cpu().detach())
        label_list.append(label.cpu().detach())

        total += data.size(0)
        correct += predicted.eq(label.long()).sum().item()        
        con_matx.add(predicted.detach(),label.detach()) 
        pbar.set_description(' acc: %.3f'% (100.* correct / total))


    recall = recall_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    precision = precision_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average=None)
    f1 = f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),average='macro')
    auc = auroc(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),task='multiclass', num_classes=args.n_classes)
    f1_micro = multiclass_f1_score(torch.cat(label_list).numpy(), torch.cat(pred_list).numpy(),num_classes=args.n_classes, average='micro')
    
    print(correct, total)
    acc = 100.* correct/total

    print('val epoch:', epoch, ' val acc: ', acc, 'recall:', recall, "precision:", precision, "f1_macro:",f1, "f1_micro:",f1_micro, "auc:",auc)
    #vis.log('epoch:{epoch},val_acc:{val_acc},val_cm:{val_cm},recall:{recall},precision:{precision},f1:{f1},f1_micro:{f1_micro},aud{auc}'.format(epoch=epoch,val_acc=acc,val_cm=str(con_matx.value()),recall=recall,precision=precision,f1=f1,f1_micro=f1_micro,auc=auc))
    wandb.log({'epoch': epoch, 'val_acc' : acc, 'val_cm' : str(con_matx.value()) ,'recall' : recall , 'precision' : precision, 'f1' : f1, 'f1_micro' : f1_micro , 'auc' : auc} ,step=epoch)

    if auc >= best_auc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'auc': auc,
            'epoch': epoch,
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_auc = auc


if __name__ == "__main__":
    main()
        

